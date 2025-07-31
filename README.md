# RealtimeUltrasoundSegmentation
Realtime Ultrasound Segmentation using the pyclariuscast api and an EfficientNet based segmentation model

# conda installation
```bash
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

```bash
cd ~/RealtimeUltrasoundSegmentation
conda env create -f ./cast-12.0.2-macos.arm64/env.yaml
```

# setup (Mac-Intel chip-x86_64)
If you are using an older, intel based mac, use the following instructions. 
In the cast-12.0.2-macos.x86_64 folder there is a libcast.dylib file that you need to copy over to your conda environment's lib folder.

the command to copy the dylib to the correct folder should look something like this:
## cp ./cast-12.0.2-macos.x86_64/libcast.dylib /opt/homebrew/Caskroom/miniconda/us/bin/lib/

# setup (Mac-Apple Silicon-ARM64)
If you are using a newer apple silicon based mac, use the following command: 
In the cast-12.0.2-macos.arm64 folder there is a libcast.dylib file that you need to copy over to your conda environment's lib folder.

the command to copy the dylib to the correct folder should look something like this:
## cp ./cast-12.0.2-macos.arm64/libcast.dylib /opt/homebrew/Caskroom/miniconda/us/bin/lib/


# installing dependencies
- all required dependencies can be installed using a conda environment using the file cast-12.0.2-macos.arm64/env.yaml: 
## conda env create -f environment.yaml

## setting up .env file
Create a .env file and store the following variables:
- BASE_DIR: an absolute folder path with the applicable operating system folder path (ex. "/Users/rahul.prabhu/RealtimeUltrasoundSegmentation/cast-12.0.2-macos.arm64")
- MODEL_PATH: the name of the model file that you are using (ex. "best_mhu.pth")

## Running Programs
- pysidecaster.py: navigate to the appropriate folder and run the following command in your terminal - "python pysidecaster.py" - this should launch the pysidecaster GUI. To record IMU and image data, first connect to the probe network by turning it on, and then looking for the network that matches the serial code on the probe - the password to the probe will be located in the scanner status? settings in the Clarius app. Once connected, start broadcasting via the Clarius app and click connect on the GUI - this will immediately start recording the IMU data and start saving image frames to a positions/images folder matching the time and date. Once done, click disconnect / quit which will save the positional data to a csv. 
- 3DstickGUI.py - navigate to the appropriate folder and run the following command in your terminal - "python 3DstickGUI.py" - this will launch another GUI containing a probe object. To show the direction of a probe for a run that was captured using pysidecaster, upload a quaternion csv file using the provided button.
- convex_hull.py - navigate to the appropriate folder and run the following command in your terminal - "python convex_hull.py". You will need to have the quaternion data in the positions folder (which should be already done assuming you have the correct BASE_DIR env var loaded) and download the appropriate images from OneDrive downloaded to the images folder https://ucsdcloud-my.sharepoint.com/:f:/r/personal/raprabhu_ucsd_edu/Documents/RealtimeUltrasoundSegmentation?csf=1&web=1&e=U2LScA.

## running the GUI
to run the gui, run the following:
for M1-4 mac users:
```bash
cd ~/RealtimeUltrasoundSegmentation/cast-12.0.2-macos.arm64
python pysidecaster.py
```

for intel mac users:
```bash
cd ~/RealtimeUltrasoundSegmentation/cast-12.0.2-macos.x86_64
python pysidecaster.py
```

