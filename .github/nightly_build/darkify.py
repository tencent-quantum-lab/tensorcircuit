from datetime import datetime
import requests


def change_version(post=""):
    datestr = datetime.now().strftime("%Y%m%d")
    datestr += post
    with open("tensorcircuit/__init__.py", "r") as f:
        r = []
        for l in f.readlines():
            if l.startswith("__version__"):
                l = l[:-2]
                l += ".dev" + datestr + '"\n'
            r.append(l)
    # __version__ = "0.2.2.dev20220706"
    with open("tensorcircuit/__init__.py", "w") as f:
        f.writelines(r)


def update_setuppy(url=None):
    if not url:
        url = "https://raw.githubusercontent.com/refraction-ray/tensorcircuit-dev/beta/.github/nightly_build/setup.py"
    r = requests.get(url)
    with open("setup.py", "w") as f:
        f.writelines(r.text)


if __name__ == "__main__":
    change_version()
    update_setuppy()
