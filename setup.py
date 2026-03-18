#!/usr/bin/env python
import subprocess
import sys

def setup():
    print("Installing setuptools...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools==65.5.0"])
    print("Setup complete!")

if __name__ == "__main__":
    setup()
