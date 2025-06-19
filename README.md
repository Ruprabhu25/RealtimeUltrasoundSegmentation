# RealtimeUltrasoundSegmentation
Realtime Ultrasound Segmentation using the pyclariuscast api and an EfficientNet based segmentation model

# Clarius Cast API: Setup & Streaming on macOS (ARM64)

This guide will walk you through downloading, setting up, and running the **Clarius Cast API** with Python on macOS (Apple Silicon), including streaming from your Clarius ultrasound device to your laptop.

---

## 🔗 Quick Links

- **Clarius Cast GitHub Repo**: [https://github.com/clariusdev/cast](https://github.com/clariusdev/cast)  
- **Latest Release Download**: [https://github.com/clariusdev/cast/releases](https://github.com/clariusdev/cast/releases)  
- **Cast API with Python**: [https://github.com/clariusdev/cast/tree/master/examples/python](https://github.com/clariusdev/cast/tree/master/examples/python)

---

## 🛠️ Setup Instructions

### 1. Download and Extract

- Go to the [releases page](https://github.com/clariusdev/cast/releases).
- Download the latest `cast-<version>-macos.arm64.zip` file.
- Unzip it to a folder, e.g.:

```bash
~/Downloads/cast-12.0.2-macos.arm64
```

---

### 2. Add Python Example Files

- From the GitHub repo, download the contents of:  
  [`examples/python`](https://github.com/clariusdev/cast/tree/master/examples/python)

- Move all these files into the unzipped release folder:
```bash
~/Downloads/cast-12.0.2-macos.arm64/
```

---

### 3. Check Python Version & Move `.so` File

- Check your Python version:
```bash
python --version
```

- Inside the `python3` subfolders, navigate to the matching version (e.g., `python312` for Python 3.12) and move the `pyclariuscast.so` file to the **main release folder**:
```bash
mv python312/pyclariuscast.so .
```

---

## 🧪 Run the Caster & View US Stream

```bash
python pysidecaster.py
```

---

## ❗ If It Doesn’t Work...

### ✅ Fix 1: Dynamic Linker Path Issue

Tell the dynamic linker to load `libcast.dylib` from the same directory where `pyclariuscast.so` is located:

```bash
cd ~/Downloads/cast-12.0.2-macos.arm64
install_name_tool -change @rpath/libcast.dylib @loader_path/libcast.dylib pyclariuscast.so
```

---

### ✅ Fix 2: macOS Gatekeeper Quarantine

Remove quarantine flags that may block execution:

```bash
xattr -d com.apple.quarantine pyclariuscast.so     
xattr -d com.apple.quarantine libcast.dylib
```

Then sign the files:

```bash
codesign --force --deep --sign - pyclariuscast.so
codesign --force --deep --sign - libcast.dylib
```

---

### ✅ Fix 3: Install Dependencies

If not already installed:

```bash
pip install PySide6
pip install pillow
```

---

## 🧭 See Device Orientation

Get IMU data (device angle, rotation, etc.):

```bash
python pyimu.py
```

---

## 📁 Final Folder Structure

Your folder should now look like this:

```
cast-12.0.2-macos.arm64/
│
├── pyclariuscast.so
├── libcast.dylib
├── pycaster.py
├── pysidecaster.py
├── pyimu.py
├── scanner.mtl
├── scanner.obj
├── ...
└── (other Python example files)
```

---

## 🎉 Connect US Clarius Device from iPad


# setup (Mac)
Use conda for your virtual environment. In the cast-12.0.2-macos.arm64 folder there is a libcast.dylib file that you need to copy over to your conda environment's lib folder.

the command to copy the dylib to the correct folder should look something like this:
## cp ./cast-12.0.2-macos.arm64/libcast.dylib /opt/homebrew/Caskroom/miniconda/NAME_OF_CONDA_ENV/bin/../lib/

# installing dependencies
- all required dependencies can be installed using a conda environment using the file cast-12.0.2-macos.arm64/env.yaml: 
## conda env create -f environment.yaml

# setup (Windows)
Use conda for your virtual environment. In the cast-12.0.2-windows.x86_64 folder there is a libcast.pyd file that you need to copy over to your conda environment's DLL folder.

the command to copy the dylib to the correct folder should look something like this:
# TODO: Add file line here

## setting up .env file
Create a .env file and store the following variables:
- BASE_DIR: an absolute folder path with the applicable operating system folder path (ex. "/Users/rahul.prabhu/..../RealtimeUltrasoundSegmentation/cast-12.0.2-macos.arm64")
- MODEL_PATH: the name of the model path file that you are using (ex. "best_mhu.pth")

## Running Programs
- pysidecaster.py: navigate to the appropriate folder and run the following command in your terminal - "python pysidecaster.py" - this should launch the pysidecaster GUI. To record IMU and image data, first connect to the probe network by turning it on, and then looking for the network that matches the serial code on the probe - the password to the probe will be located in the scanner status? settings in the Clarius app. Once connected, start broadcasting via the Clarius app and click connect on the GUI - this will immediately start recording the IMU data and start saving image frames to a positions/images folder matching the time and date. Once done, click disconnect / quit which will save the positional data to a csv. 
- 3DstickGUI.py - navigate to the appropriate folder and run the following command in your terminal - "python 3DstickGUI.py" - this will launch another GUI containing a probe object. To show the direction of a probe for a run that was captured using pysidecaster, upload a quaternion csv file using the provided button.
- convex_hull.py - navigate to the appropriate folder and run the following command in your terminal - "python convex_hull.py". You will need to have the quaternion data in the positions folder (which should be already done assuming you have the correct BASE_DIR env var loaded) and download the appropriate images from OneDrive downloaded to the images folder https://ucsdcloud-my.sharepoint.com/:f:/r/personal/raprabhu_ucsd_edu/Documents/RealtimeUltrasoundSegmentation?csf=1&web=1&e=U2LScA.
