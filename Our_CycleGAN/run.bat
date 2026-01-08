cd /d %~dp0
cd ..

call conda activate proj

python Our_CycleGAN/main.py

pause