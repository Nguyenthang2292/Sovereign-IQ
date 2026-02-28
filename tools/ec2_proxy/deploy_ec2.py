"""
EC2 Proxy Auto-Deploy Script
=============================
Tự động:
1. Tạo EC2 t2.micro (Free Tier) tại ap-southeast-1 (Singapore)
2. Cấp Elastic IP (IP tĩnh vĩnh viễn)
3. Lưu cấu hình vào .env
4. In ra IP cần whitelist trên Binance

Usage:
    python tools/ec2_proxy/deploy_ec2.py           # Deploy mới
    python tools/ec2_proxy/deploy_ec2.py --status  # Kiểm tra instance hiện tại
    python tools/ec2_proxy/deploy_ec2.py --destroy # Xóa instance + giải phóng EIP
"""

import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# ─── CONFIG ──────────────────────────────────────────────────────────────────
REGION = "ap-southeast-1"  # Singapore — gần Binance nhất
INSTANCE_TYPE = "t2.micro"  # Free Tier eligible
KEY_NAME = "binance-proxy-key"
SG_NAME = "binance-proxy-sg"
TAG_NAME = "binance-proxy"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KEY_FILE = PROJECT_ROOT / "tools" / "ec2_proxy" / f"{KEY_NAME}.pem"
ENV_FILE = PROJECT_ROOT / "modules" / "auto_trade" / ".env"
# ─────────────────────────────────────────────────────────────────────────────


def get_boto3_session() -> boto3.Session:
    """Tạo boto3 session, dùng AWS credentials đã có sẵn trong project."""
    # Load .env để lấy AWS credentials nếu có
    _load_env()
    session = boto3.Session(region_name=REGION)
    # Verify credentials
    try:
        session.client("sts").get_caller_identity()
    except Exception as e:
        print(f"❌ AWS credentials not found: {e}")
        print("   Hãy đảm bảo AWS_ACCESS_KEY_ID và AWS_SECRET_ACCESS_KEY có trong .env")
        sys.exit(1)
    return session


def _load_env() -> None:
    """Load .env files để lấy AWS credentials."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / "modules" / "auto_trade" / ".env", override=False)
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass


def get_latest_amazon_linux_ami(ec2) -> str:
    """Lấy AMI ID mới nhất của Amazon Linux 2023 cho ap-southeast-1."""
    response = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": ["al2023-ami-2023*-x86_64"]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    )
    images = sorted(response["Images"], key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError("Không tìm thấy Amazon Linux 2023 AMI")
    ami_id = images[0]["ImageId"]
    print(f"   AMI: {ami_id} ({images[0]['Name']})")
    return ami_id


def create_key_pair(ec2) -> str:
    """Tao key pair va luu file .pem ve local."""
    # Xoa key cu neu ton tai tren AWS
    try:
        ec2.delete_key_pair(KeyName=KEY_NAME)
    except ClientError:
        pass

    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if KEY_FILE.exists():
        # Force-delete on Windows — key file may be locked (read-only / restricted ACL)
        try:
            import os as _os

            _os.chmod(str(KEY_FILE), 0o777)
        except Exception:
            pass
        if sys.platform == "win32":
            # Remove read-only attribute and reset ACL
            subprocess.run(["attrib", "-R", str(KEY_FILE)], capture_output=True)
            subprocess.run(["icacls", str(KEY_FILE), "/reset"], capture_output=True)
            subprocess.run(
                ["icacls", str(KEY_FILE), "/grant", f"{os.environ.get('USERNAME', 'Admin')}:F"],
                capture_output=True,
            )
        KEY_FILE.unlink(missing_ok=True)

    response = ec2.create_key_pair(KeyName=KEY_NAME)
    pem_content = response["KeyMaterial"]

    KEY_FILE.write_text(pem_content)
    # chmod 400 -- read-only
    KEY_FILE.chmod(stat.S_IRUSR)
    if sys.platform == "win32":
        _fix_windows_key_permissions(str(KEY_FILE))

    print(f"   Key saved: {KEY_FILE}")
    return KEY_NAME


def _fix_windows_key_permissions(key_path: str) -> None:
    """Fix SSH key permissions on Windows (icacls)."""
    username = os.environ.get("USERNAME", "Admin")
    subprocess.run(["icacls", key_path, "/inheritance:r"], capture_output=True)
    subprocess.run(["icacls", key_path, "/grant:r", f"{username}:(R)"], capture_output=True)


def get_or_create_security_group(ec2) -> str:
    """Tạo Security Group cho SSH tunnel proxy."""
    # Kiểm tra SG đã tồn tại
    try:
        response = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [SG_NAME]}])
        if response["SecurityGroups"]:
            sg_id = response["SecurityGroups"][0]["GroupId"]
            print(f"   Security Group (existing): {sg_id}")
            return sg_id
    except ClientError:
        pass

    # Tạo mới
    response = ec2.create_security_group(
        GroupName=SG_NAME,
        Description="Binance Proxy SSH Tunnel - allow SSH inbound",
    )
    sg_id = response["GroupId"]

    # Inbound: SSH từ bất kỳ IP (local IP thay đổi nên allow all)
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH for SOCKS5 tunnel"}],
            }
        ],
    )
    print(f"   Security Group (created): {sg_id}")
    return sg_id


def launch_ec2_instance(ec2, ami_id: str, sg_id: str) -> str:
    """Launch EC2 instance với user data để enable SSH TCP forwarding."""
    user_data = """#!/bin/bash
# Enable TCP forwarding for SOCKS5 proxy
echo "AllowTcpForwarding yes" >> /etc/ssh/sshd_config
echo "ClientAliveInterval 60" >> /etc/ssh/sshd_config
echo "ClientAliveCountMax 10" >> /etc/ssh/sshd_config
systemctl restart sshd

# Log public IP
curl -s https://api.ipify.org > /tmp/public_ip.txt
"""
    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        SecurityGroupIds=[sg_id],
        MinCount=1,
        MaxCount=1,
        UserData=user_data,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": TAG_NAME},
                    {"Key": "Purpose", "Value": "binance-proxy"},
                ],
            }
        ],
    )
    instance_id = response["Instances"][0]["InstanceId"]
    print(f"   Instance ID: {instance_id}")
    return instance_id


def wait_for_instance(ec2, instance_id: str) -> str:
    """Đợi instance running và lấy public IP tạm."""
    print("   Waiting for instance to start", end="", flush=True)
    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    print(" ✅")

    response = ec2.describe_instances(InstanceIds=[instance_id])
    instance = response["Reservations"][0]["Instances"][0]
    return instance.get("PublicIpAddress", "")


def allocate_elastic_ip(ec2) -> tuple[str, str]:
    """Allocate một Elastic IP mới."""
    response = ec2.allocate_address(Domain="vpc")
    eip = response["PublicIp"]
    alloc_id = response["AllocationId"]
    print(f"   Elastic IP allocated: {eip} (AllocationId: {alloc_id})")
    return eip, alloc_id


def associate_elastic_ip(ec2, instance_id: str, alloc_id: str) -> None:
    """Associate Elastic IP với instance."""
    ec2.associate_address(InstanceId=instance_id, AllocationId=alloc_id)
    print("   Elastic IP associated ✅")


def update_env_file(elastic_ip: str, instance_id: str, alloc_id: str) -> None:
    """Thêm EC2 proxy config vào .env file."""
    env_content = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else ""

    # Xóa section cũ nếu tồn tại
    lines = env_content.splitlines()
    filtered = []
    skip = False
    for line in lines:
        if line.strip() == "# ============================================":
            if skip:
                skip = False
                continue
        if "EC2 Proxy" in line or line.startswith("EC2_PROXY_"):
            skip = True
            continue
        if not skip:
            filtered.append(line)

    # Use forward slashes for path — avoids dotenv interpreting \N as Unicode escape
    key_path_fwd = str(KEY_FILE).replace("\\", "/")
    new_section = f"""
# ============================================
# EC2 Proxy - Fixed IP cho Binance API
# IP tinh: {elastic_ip} (Elastic IP, khong bao gio thay doi)
# Whitelist IP nay tren Binance API Management
# ============================================
EC2_PROXY_ENABLED=true
EC2_PROXY_HOST={elastic_ip}
EC2_PROXY_INSTANCE_ID={instance_id}
EC2_PROXY_ALLOC_ID={alloc_id}
EC2_PROXY_KEY_PATH={key_path_fwd}
EC2_PROXY_USER=ec2-user
EC2_PROXY_PORT=1080
"""
    updated = "\n".join(filtered) + new_section
    ENV_FILE.write_text(updated, encoding="utf-8")
    print(f"   .env updated: {ENV_FILE}")


def find_existing_instance(ec2) -> dict | None:
    """Tìm instance binance-proxy đang chạy."""
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": [TAG_NAME]},
            {"Name": "instance-state-name", "Values": ["running", "stopped", "pending"]},
        ]
    )
    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            return instance
    return None


def find_elastic_ip(ec2) -> dict | None:
    """Tìm Elastic IP đã allocate cho proxy."""
    response = ec2.describe_addresses(Filters=[{"Name": "tag:Purpose", "Values": ["binance-proxy"]}])
    if response["Addresses"]:
        return response["Addresses"][0]
    return None


def cmd_deploy(ec2) -> None:
    """Deploy EC2 proxy từ đầu."""
    print("\n🚀 Deploying Binance Proxy EC2...")

    # Check nếu đã tồn tại
    existing = find_existing_instance(ec2)
    if existing:
        instance_id = existing["InstanceId"]
        state = existing["State"]["Name"]
        print(f"\n⚠️  Instance đã tồn tại: {instance_id} (state: {state})")
        answer = input("Tiếp tục và tạo mới sẽ xóa instance cũ. Tiếp tục? (y/N): ")
        if answer.lower() != "y":
            print("Hủy.")
            return
        print(f"Terminating old instance {instance_id}...")
        ec2.terminate_instances(InstanceIds=[instance_id])
        ec2.get_waiter("instance_terminated").wait(InstanceIds=[instance_id])

    print("\n[1/6] Finding latest Amazon Linux 2023 AMI...")
    ami_id = get_latest_amazon_linux_ami(ec2)

    print("\n[2/6] Creating SSH key pair...")
    create_key_pair(ec2)

    print("\n[3/6] Creating Security Group...")
    sg_id = get_or_create_security_group(ec2)

    print("\n[4/6] Launching EC2 instance...")
    instance_id = launch_ec2_instance(ec2, ami_id, sg_id)

    print("\n[5/6] Waiting for instance to start...")
    wait_for_instance(ec2, instance_id)

    print("\n[6/6] Allocating & associating Elastic IP...")
    elastic_ip, alloc_id = allocate_elastic_ip(ec2)

    # Tag EIP
    ec2.create_tags(
        Resources=[alloc_id],
        Tags=[{"Key": "Purpose", "Value": "binance-proxy"}],
    )

    associate_elastic_ip(ec2, instance_id, alloc_id)

    print("\n[7/7] Saving config to .env...")
    update_env_file(elastic_ip, instance_id, alloc_id)

    print("\n" + "=" * 60)
    print("✅ DEPLOYMENT COMPLETE!")
    print("=" * 60)
    print(f"\n📌 ELASTIC IP (cố định mãi mãi): {elastic_ip}")
    print(f"   Instance ID: {instance_id}")
    print(f"   Key file   : {KEY_FILE}")
    print()
    print("🔑 BƯỚC TIẾP THEO:")
    print(f"   1. Truy cập Binance → API Management → Whitelist IP: {elastic_ip}")
    print("   2. Đợi ~60s để EC2 khởi động xong SSH")
    print("   3. Khởi động lại app — tunnel sẽ tự động kết nối")

    print()
    print("📋 Test ngay:")
    print("   python tools/ec2_proxy/check_proxy.py")
    print("=" * 60)


def cmd_status(ec2) -> None:
    """Hiển thị trạng thái instance hiện tại."""
    print("\n📊 Binance Proxy Status")
    print("-" * 40)
    instance = find_existing_instance(ec2)
    if instance:
        state = instance["State"]["Name"]
        instance_id = instance["InstanceId"]
        public_ip = instance.get("PublicIpAddress", "N/A")
        print(f"Instance ID : {instance_id}")
        print(f"State       : {state}")
        print(f"Public IP   : {public_ip}")
        print(f"Type        : {instance['InstanceType']}")
        print(f"Region      : {REGION}")
    else:
        print("❌ Không tìm thấy instance binance-proxy")

    eip = find_elastic_ip(ec2)
    if eip:
        print(f"\nElastic IP  : {eip['PublicIp']}")
        print(f"Alloc ID    : {eip['AllocationId']}")
        assoc = eip.get("AssociationId", "Not associated")
        print(f"Associated  : {assoc}")
    else:
        print("\nElastic IP : Not found")


def cmd_destroy(ec2) -> None:
    """Xóa instance và giải phóng Elastic IP."""
    print("\n⚠️  DESTROY Binance Proxy (sẽ mất IP hiện tại)")
    answer = input("Bạn chắc chắn? Nhập 'yes' để xác nhận: ")
    if answer != "yes":
        print("Hủy.")
        return

    instance = find_existing_instance(ec2)
    if instance:
        instance_id = instance["InstanceId"]
        print(f"Terminating {instance_id}...")
        ec2.terminate_instances(InstanceIds=[instance_id])
        print("Waiting for termination...")
        ec2.get_waiter("instance_terminated").wait(InstanceIds=[instance_id])
        print("✅ Instance terminated")

    eip = find_elastic_ip(ec2)
    if eip:
        alloc_id = eip["AllocationId"]
        if "AssociationId" in eip:
            ec2.disassociate_address(AssociationId=eip["AssociationId"])
        ec2.release_address(AllocationId=alloc_id)
        print(f"✅ Elastic IP {eip['PublicIp']} released")

    print("\n✅ Destroy complete. Config sẽ cần update lại nếu deploy mới.")


def main():
    parser = argparse.ArgumentParser(description="Binance Proxy EC2 Manager")
    parser.add_argument("--status", action="store_true", help="Show current instance status")
    parser.add_argument("--destroy", action="store_true", help="Terminate instance and release EIP")
    args = parser.parse_args()

    session = get_boto3_session()
    ec2 = session.client("ec2", region_name=REGION)

    if args.status:
        cmd_status(ec2)
    elif args.destroy:
        cmd_destroy(ec2)
    else:
        cmd_deploy(ec2)


if __name__ == "__main__":
    main()
