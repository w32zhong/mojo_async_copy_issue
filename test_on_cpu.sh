set -ex
mojo build foo.mojo --emit shared-lib -o bar.so
python run_on_cpu.py
rm -f bar.so
