# About

Simple demo to run Alpamayo on AWS EC2 (g5.4xlarge instance)

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
cd aws

Region=ap-northeast-1
# AvailabilityZone=ap-northeast-1a
# ImageId=ami-0f65fc8c24ec8d2a1  # Ubuntu Server 24.04
# InstanceType=t3.medium
AvailabilityZone=ap-northeast-1d
ImageId=ami-0e7d0c8815f409923   # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Ubuntu 24.04)
InstanceType=g5.4xlarge
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

# Acknowledgements
- Drive Video by Dashcam Roadshow
- 4K東京ドライブ：晴海→東京駅→スカイツリー 11km 
  - https://www.youtube.com/watch?v=ZZjDfYQQb0c
- 4K 東京ドライブ  豊洲→渋谷→品川→秋葉原
  - https://www.youtube.com/watch?v=exouyX15boM
