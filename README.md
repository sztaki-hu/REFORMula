The scientific background of the ViSoS simulator can be found in the attached pfd files, i.e., REFORMula_simulator_HU.pdf (in Hungarian, that is the original version) and REFORMula_simulator_EN.pdf (in English).
## Install
```
git clone https://github.com/sztaki-hu/REFORMula.git
cd REFORMula
python -m venv .venv
```
### Activate Python virtual environment
Windows CommandLine
```
.venv\Scripts\activate.bat
```
Windows PowerShell
```
.venv\Scripts\Activate.ps1
```
Linux and Mac
```
source ./.venv/bin/activate
```
### Upgrade pip and install required packages
```
py -m pip install --upgrade pip
pip install -r requirements.txt
```
## Edit motion.py
Search for ```PLACE_YOUR_TOKEN``` and replace it with your own token. Save and close.
## Run
```python motion.py```
