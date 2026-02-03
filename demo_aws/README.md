# Simple demo to run Alpamayo on AWS EC2

<img width="1160" height="540" alt="output" src="https://github.com/user-attachments/assets/3caaa728-1169-42ab-8131-41b44ba0b2cd" />

![output_conv mp4_snapshot_00 04 540](https://github.com/user-attachments/assets/828cf610-7258-4059-ba21-261ed66aeee2)

https://github.com/user-attachments/assets/821a12af-d123-40c4-965d-88537f45f3a0


- demo_01_example_clip.py
  - A simple demo based on the original test_inference.py
  - Reduction of input image size and Use only the front camera image
  - Visualization of trajectory outputs
  - Overlaying trajectories onto the input images
- demo_02_from_image_folder.py
  - A demo using images from a folder as input
- demo_03_from_video.py
  - A demo with MP4 video input and video output

```bash
source ar1_venv/bin/activate

python3 demo_aws/demo_01_example_clip.py
python3 demo_aws/demo_02_from_image_folder.py
python3 demo_aws/demo_03_from_video.py
```

## Build EC2 Server

```bash
cd demo_aws

Region=ap-northeast-1
# AvailabilityZone=ap-northeast-1a
# ImageId=ami-0f65fc8c24ec8d2a1  # Ubuntu Server 24.04
# InstanceType=t3.medium
AvailabilityZone=ap-northeast-1d
ImageId=ami-0e7d0c8815f409923   # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Ubuntu 24.04)
InstanceType=g6.2xlarge
# InstanceType=p5.4xlarge
RootVolumeSize=256

SystemName=test-alpamayo
TemplateFileName=./ec2_public_alb.yaml

aws cloudformation deploy \
--region "${Region}" \
--stack-name "${SystemName}" \
--template-file ${TemplateFileName} \
--capabilities CAPABILITY_NAMED_IAM \
--parameter-overrides \
SystemName="${SystemName}" \
AvailabilityZone="${AvailabilityZone}" \
ImageId="${ImageId}" \
InstanceType="${InstanceType}" \
RootVolumeSize="${RootVolumeSize}"
```

## SSH Configuration

- Configure `~/.ssh/config`
  - (Optional) For Windows: Replace the followings
    - `sh -c` -> `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`
    - `&&` -> `;`
- Connect

```bash
# ~/.ssh/config
Host i-* mi-*
    ProxyCommand sh -c "aws ec2-instance-connect send-ssh-public-key --instance-id %h --instance-os-user %r --ssh-public-key 'file://~/.ssh/id_rsa.pub' && aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters 'portNumber=%p'"

# (Optional) To specify host id
Host test-ec2-server
    HostName i-00000000000000000
    User ubuntu
    # User ec2-user
    ProxyCommand sh -c "aws ec2-instance-connect send-ssh-public-key --instance-id %h --instance-os-user %r --ssh-public-key 'file://~/.ssh/id_rsa.pub' && aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters 'portNumber=%p'"
```

```bash
ssh ubuntu@i-00000000000000000
ssh ec2-user@i-00000000000000000
# or
ssh test-ec2-server
```

## Setup

```bash
sudo apt update
sudo apt install -y python3-pip
sudo apt install -y nvidia-cuda-toolkit

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/iwatake2222/alpamayo-demo-aws.git
cd alpamayo-demo-aws
uv venv ar1_venv
source ar1_venv/bin/activate
uv sync --active

pip install huggingface_hub --break-system-packages
huggingface-cli login

python3 demo_aws/demo_01_example_clip.py
```

# Acknowledgements
- Drive Video by Dashcam Roadshow
- 4K Tokyo Scenic Drive: Bayside to Tokyo Station and Skytree 11km
  - https://www.youtube.com/watch?v=ZZjDfYQQb0c
- Tokyo Drive 4K | Toyosu - Shibuya - Shinagawa - Akihabara
  - https://www.youtube.com/watch?v=exouyX15boM
