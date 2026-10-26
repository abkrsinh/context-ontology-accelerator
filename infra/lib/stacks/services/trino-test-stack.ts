// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as elbv2 from "aws-cdk-lib/aws-elasticloadbalancingv2";
import * as elbv2Targets from "aws-cdk-lib/aws-elasticloadbalancingv2-targets";
import * as iam from "aws-cdk-lib/aws-iam";
import * as route53 from "aws-cdk-lib/aws-route53";
import * as targets from "aws-cdk-lib/aws-route53-targets";
import * as acm from "aws-cdk-lib/aws-certificatemanager";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import { Construct } from "constructs";
import { SCLStack } from "../../constructs/scl-stack";
import { resolveContext } from "../../context";
import { NetworkStack } from "../foundation/network-stack";

export interface TrinoTestStackProps extends cdk.StackProps {
  readonly network: NetworkStack;
  /** ACM cert for `hostname`. Must be publicly trusted — see the class doc. */
  readonly certificateArn: string;
  /** S3 Tables table-bucket ARN, used as the Iceberg REST warehouse. */
  readonly tableBucketArn: string;
  readonly hostname: string;
  readonly hostedZoneId: string;
  readonly hostedZoneName: string;
  /** AZ for the coordinator; m7g capacity is not uniform across us-east-1. */
  readonly instanceAvailabilityZone: string;
  /** Extra client SGs allowed to the ALB, e.g. the serve runtime's AgentCore SG.
   *  Taken as ids rather than imported, so the product stacks export nothing for
   *  test infrastructure. */
  readonly extraClientSecurityGroupIds?: string[];
  /** Namespace the credential secret is tagged for, so discovery may read it. */
  readonly namespaceId?: string;
}

/**
 * A single-node Trino coordinator over the S3 Tables Iceberg REST catalog, for
 * exercising the TRINO source type end to end.
 *
 * Opt-in via `-c trino_test_enabled=true`; nothing is created otherwise. This is
 * test infrastructure and no COA stack depends on it.
 *
 * Three constraints drove the shape:
 *   - COA's Lambda SG egresses on 443 only, so the coordinator answers on 443.
 *   - The Python Trino client verifies TLS against certifi, so the cert must be
 *     publicly trusted — hence an ALB with an ACM cert rather than Trino's own
 *     self-signed TLS.
 *   - `TrinoAuthMode` has no "none", so the coordinator runs a password
 *     authenticator seeded from a generated secret.
 *
 * Trino runs as the official container image, so the required JDK is the image's
 * concern. The ALB is internal and its only ingress is COA's Lambda SG.
 */
export class TrinoTestStack extends SCLStack {
  public readonly endpoint: string;
  public readonly credentialSecret: secretsmanager.Secret;

  constructor(scope: Construct, id: string, props: TrinoTestStackProps) {
    super(scope, id, props);
    this.addComponentTag("trino-test");

    const vpc = props.network.vpc;

    // Generated, never written to the repo or to a config file on the instance
    // in plaintext form beyond the htpasswd hash user-data computes.
    this.credentialSecret = new secretsmanager.Secret(this, "TrinoCredential", {
      secretName: this.prefixed("trino-test-credential"),
      generateSecretString: {
        secretStringTemplate: JSON.stringify({ username: "coa" }),
        generateStringKey: "password",
        excludePunctuation: true,
        passwordLength: 24,
      },
    });
    // COA's discovery role reads secrets gated on this tag existing, so without
    // it a scan cannot fetch the credential.
    if (props.namespaceId) {
      cdk.Tags.of(this.credentialSecret).add(
        `${resolveContext(this.node).prefix}:namespace`,
        props.namespaceId,
      );
    }

    const albSg = new ec2.SecurityGroup(this, "AlbSg", {
      vpc,
      securityGroupName: this.prefixed("trino-test-alb-sg"),
      description: "Trino test ALB",
      allowAllOutbound: false,
    });
    const trinoSg = new ec2.SecurityGroup(this, "TrinoSg", {
      vpc,
      securityGroupName: this.prefixed("trino-test-sg"),
      description: "Trino test coordinator",
      // Needs egress for the S3 Tables REST API, S3 data and the container pull.
      allowAllOutbound: true,
    });

    // SG-to-SG, never a CIDR: only COA's Lambdas reach the ALB, only the ALB
    // reaches Trino. Nothing is internet-reachable.
    albSg.addIngressRule(
      props.network.lambdaSecurityGroup,
      ec2.Port.tcp(443),
      "COA Lambdas (discovery + sources API) to Trino",
    );
    (props.extraClientSecurityGroupIds ?? []).forEach((sgId, i) => {
      albSg.addIngressRule(
        ec2.Peer.securityGroupId(sgId),
        ec2.Port.tcp(443),
        `Additional COA client ${i} to Trino`,
      );
    });
    albSg.addEgressRule(trinoSg, ec2.Port.tcp(8080), "ALB to Trino HTTP");
    trinoSg.addIngressRule(albSg, ec2.Port.tcp(8080), "ALB to Trino HTTP");

    const role = new iam.Role(this, "TrinoRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      managedPolicies: [
        // ReadOnly, not FullAccess: Trino only reads the catalog and table data.
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "AmazonS3TablesReadOnlyAccess",
        ),
        // Session Manager, so the instance needs no key pair and no SSH ingress.
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "AmazonSSMManagedInstanceCore",
        ),
      ],
    });
    this.credentialSecret.grantRead(role);

    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      "set -eux",
      "dnf install -y docker httpd-tools jq",
      "systemctl enable --now docker",
      "mkdir -p /etc/trino/catalog",
      // TLS terminates at the ALB, so Trino serves plain HTTP. Password auth over
      // HTTP requires process-forwarded, and is safe only because 8080 admits the
      // ALB's SG alone.
      `cat > /etc/trino/config.properties <<'CFG'
coordinator=true
node-scheduler.include-coordinator=true
http-server.http.port=8080
discovery.uri=http://localhost:8080
http-server.authentication.type=PASSWORD
http-server.process-forwarded=true
# Required whenever authentication is enabled, even single-node; Trino refuses to
# start without it. Generated per boot and never needed off-box.
internal-communication.shared-secret=__SHARED_SECRET__
CFG`,
      "sed -i \"s|__SHARED_SECRET__|$(openssl rand -base64 32 | tr -d '\\n')|\" /etc/trino/config.properties",
      "echo node.environment=test > /etc/trino/node.properties",
      `cat > /etc/trino/jvm.config <<'CFG'
-server
-Xmx6G
-XX:+UseG1GC
-XX:+ExitOnOutOfMemoryError
-Djdk.attach.allowAttachSelf=true
CFG`,
      `cat > /etc/trino/password-authenticator.properties <<'CFG'
password-authenticator.name=file
file.password-file=/etc/trino/password.db
CFG`,
      // SigV4 as the 's3tables' service; credentials come from the instance role,
      // so no key material reaches the config.
      `cat > /etc/trino/catalog/iceberg.properties <<'CFG'
connector.name=iceberg
iceberg.catalog.type=rest
iceberg.rest-catalog.uri=https://s3tables.${this.region}.amazonaws.com/iceberg
iceberg.rest-catalog.warehouse=${props.tableBucketArn}
# SIGV4 requires Trino >= 477; 476 accepts only NONE and OAUTH2.
iceberg.rest-catalog.security=SIGV4
iceberg.rest-catalog.signing-name=s3tables
fs.native-s3.enabled=true
s3.region=${this.region}
CFG`,
      `PW=$(aws secretsmanager get-secret-value --region ${this.region} --secret-id ${this.credentialSecret.secretArn} --query SecretString --output text | jq -r .password)`,
      'htpasswd -bBC 10 -c /etc/trino/password.db coa "$PW"',
      "docker run -d --restart=always --name trino -p 8080:8080 -v /etc/trino:/etc/trino trinodb/trino:484",
    );

    const instance = new ec2.Instance(this, "Coordinator", {
      vpc,
      // AZ-pinned: m7g has no capacity in every us-east-1 AZ, and an unpinned
      // instance lands in whichever subnet is first.
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
        availabilityZones: [props.instanceAvailabilityZone],
      },
      instanceType: ec2.InstanceType.of(
        ec2.InstanceClass.M7G,
        ec2.InstanceSize.XLARGE,
      ),
      machineImage: ec2.MachineImage.latestAmazonLinux2023({
        cpuType: ec2.AmazonLinuxCpuType.ARM_64,
      }),
      securityGroup: trinoSg,
      role,
      userData,
      instanceName: this.prefixed("trino-test"),
    });

    const alb = new elbv2.ApplicationLoadBalancer(this, "Alb", {
      vpc,
      internetFacing: false,
      securityGroup: albSg,
      loadBalancerName: this.prefixed("trino-test-alb"),
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
    });
    const listener = alb.addListener("Https", {
      port: 443,
      protocol: elbv2.ApplicationProtocol.HTTPS,
      certificates: [
        acm.Certificate.fromCertificateArn(this, "Cert", props.certificateArn),
      ],
    });
    listener.addTargets("Coordinator", {
      port: 8080,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targets: [new elbv2Targets.InstanceTarget(instance, 8080)],
      healthCheck: {
        // Unauthenticated, unlike every other Trino endpoint, so the health check
        // needs no credential.
        path: "/v1/info",
        interval: cdk.Duration.seconds(30),
      },
    });

    new route53.ARecord(this, "AliasRecord", {
      zone: route53.HostedZone.fromHostedZoneAttributes(this, "Zone", {
        hostedZoneId: props.hostedZoneId,
        zoneName: props.hostedZoneName,
      }),
      recordName: props.hostname,
      target: route53.RecordTarget.fromAlias(
        new targets.LoadBalancerTarget(alb),
      ),
    });

    this.endpoint = `https://${props.hostname}`;
    new cdk.CfnOutput(this, "TrinoEndpoint", { value: this.endpoint });
    new cdk.CfnOutput(this, "TrinoCredentialSecretArn", {
      value: this.credentialSecret.secretArn,
    });
    new cdk.CfnOutput(this, "TrinoAllowedHostsValue", {
      value: props.hostname,
      description: "Set SCL_TRINO_ALLOWED_HOSTS to this before redeploying COA",
    });
  }
}
