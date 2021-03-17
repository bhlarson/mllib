[Developer Kit](https://developer.nvidia.com/jetson-agx-xavier-developer-kit-user-guide)

[Update Jetpack with Packet Manager](https://docs.nvidia.com/jetson/jetpack/install-jetpack/index.html#package-management-tool)

[Install with SDK Manager](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html)

[Upgrade with package manager](https://docs.nvidia.com/jetson/jetpack/install-jetpack/index.html#upgrade-jetpack)

<img src="../img/AGXDevKitBottomView.png" width="400">
<img src="../img/AGXUSBConnection.png" width="400">


```console
# Find current jetpack release
$ cat /etc/nv_tegra_release
# Set to desired version.  Now r32.5.  Is this step necessary?
$ sudo nano /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
# Update package manager
$ sudo apt update
$ sudo apt dist-upgrade -y
$ sudo apt install nvidia-jetpack -y
$ sudo apt autoremove
# Find current jetpack release
$ cat /etc/nv_tegra_release
```