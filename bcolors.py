import sys

# dirty code
sys.stdout.reconfigure(encoding="utf-8")

class bcolors:
    _blue = '\033[94m'
    _cyan = '\033[96m'
    _green = '\033[32m'
    _lightgreen = '\033[92m'
    _magenta = '\033[95m'
    _red = '\033[91m'
    _yellow = '\033[93m'
    _darkyellow = '\033[33m'
    _reset = '\033[0m'

    _bold = '\033[1m'
    _underline = '\033[4m'

    @staticmethod
    def blue(t: str) -> str: return bcolors._blue + t + bcolors._reset
    @staticmethod
    def cyan(t: str) -> str: return bcolors._cyan + t + bcolors._reset
    @staticmethod
    def green(t: str) -> str: return bcolors._green + t + bcolors._reset
    @staticmethod
    def lightgreen(t: str) -> str: return bcolors._lightgreen + t + bcolors._reset
    @staticmethod
    def magenta(t: str) -> str: return bcolors._magenta + t + bcolors._reset
    @staticmethod
    def red(t: str) -> str: return bcolors._red + t + bcolors._reset
    @staticmethod
    def yellow(t: str) -> str: return bcolors._yellow + t + bcolors._reset
    @staticmethod
    def darkyellow(t: str) -> str: return bcolors._darkyellow + t + bcolors._reset
    @staticmethod
    def reset(t: str) -> str: return bcolors._reset + t + bcolors._reset

    @staticmethod
    def bold(t: str) -> str: return bcolors._bold + t + bcolors._reset
    @staticmethod
    def underline(t: str) -> str: return bcolors._underline + t + bcolors._reset
