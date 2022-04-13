"""
Some ascii art from https://www.asciiart.eu/, have fun!
"""
# pylint: disable=invalid-name

import hashlib
import sys
from typing import Any, Dict, Optional

import numpy as np

visible = False
__passwd = {"ce25e1867bc26a3209ab7f76d752fbbd"}
thismodule = sys.modules[__name__]


# def shifttext(inputs: str, shift: int) -> str:
# return "".join(chr((ord(char) + shift)) for char in inputs)


class Art:
    def __init__(self, s: str) -> None:
        self.__s = s

    def __str__(self) -> str:
        global visible
        if visible:
            return self.__s
        else:
            return ""

    __repr__ = __str__


__bike__ = """
o__
 ,>/_
(*)`(*)
"""

__bigbike__ = """
                                          $"   *.      
              d$$$$$$$P"                  $    J
                  ^$.                     4r  "
                  d"b                    .db
                 P   $                  e" $
        ..ec.. ."     *.              zP   $.zec..
    .^        3*b.     *.           .P" .@"4F      "4
  ."         d"  ^b.    *c        .$"  d"   $         %
 /          P      $.    "c      d"   @     3r         3
4        .eE........$r===e$$$$eeP    J       *..        b
$       $$$$$       $   4$$$$$$$     F       d$$$.      4
$       $$$$$       $   4$$$$$$$     L       *$$$"      4
4         "      ""3P ===$$$$$$"     3                  P
 *                 $       ""         b                J
  ".             .P                    %.             @
    %.         z*"                      ^%.        .r"
       "*==*""                             ^"*==*""   Gilo94'
"""

__moon__ = """
         ___---___                    
      .--         --.      
    ./   ()      .-. \.
   /   o    .   (   )  \ 
  / .            '-'    \         
 | ()    .  O         .  |      
|                         |      
|    o           ()       |
|       .--.          O   |            
 | .   |    |            |
  \    `.__.'    o   .  /    
   \                   /                   
    `\  o    ()      /' JT/jgs         
      `--___   ___--'
            ---
"""


__cat__ = """
 ,_     _
 |\ _,-~/
 / _  _ |    ,--.
(  @  @ )   / ,-'
 \  _T_/-._( (
 /         `. \ 
|         _  \ |
 \ \ ,  /      |
  || |-_\__   /
 ((_/`(____,-'
"""

__moonlanding__ = """
                     _  _     ____________.--.
                  |\|_|//_.-"" .'    \   /|  |
                  |.-"-"-.|   /       \_/ |  |
                  \  ||  /| __\_____________ |
                  _\_||_/_| .-""            ""-.  __
                .' '.    \//                    ".\/
                ||   '. >()_                     |()<
                ||__.-' |/\ \                    |/\ 
                   |   / "|  \__________________/.""
                  /   //  | / \ "-.__________/  /\ 
               ___|__/_|__|/___\___".______//__/__\ 
              /|\     [____________] \__/         |\ 
             //\ \     |  |=====| |   /\\         |\\
            // |\ \    |  |=====| |   | \\        | \\        ____...____....----
          .//__| \ \   |  |=====| |   | |\\       |--\\---"-""     .            ..
_____....-//___|  \_\  |  |=====| |   |_|_\\      |___\\    .                 ...'
 .      .//-.__|_______|__|_____|_|_____[__\\_____|__.-\\      .     .    ...::
        //        //        /          \ `-_\\/         \\          .....:::
  -... //     .  / /       /____________\    \\       .  \ \     .            .
      //   .. .-/_/-.                 .       \\        .-\_\-.                 .
     / /      '-----'           .             \ \      '._____.'         .
  .-/_/-.         .                          .-\_\-.                          ...
 '._____.'                            .     '._____.'                       .....
        .                                                             ...... ..
    .            .                  .                        .
   ...                    .                      .                       .      .
        ....     .                       .                    ....
 JRO      ......           . ..                       ......'
             .......             '...              ....
                                   ''''''      .              .

"""

__moonshot__ = """
                         .-.
                        ( (
                         `-'






                    .   ,- To the Moon %s !
                   .'.
                   |o|
                  .'o'.
                  |.-.|
                  '   '
                   ( )
                    )
                   ( )

               ____
          .-'""p 8o""`-.
       .-'8888P'Y.`Y[ ' `-.
     ,']88888b.J8oo_      '`.
   ,' ,88888888888["        Y`.
  /   8888888888P            Y8\ 
 /    Y8888888P'             ]88\ 
:     `Y88'   P              `888:
:       Y8.oP '- >            Y88:
|          `Yb  __             `'|
:            `'d8888bo.          :
:             d88888888ooo.      ;
 \            Y88888888888P     /
  \            `Y88888888P     /
   `.            d88888P'    ,'
     `.          888PP'    ,'
       `-.      d8P'    ,-'   -CJ-
          `-.,,_'__,,.-'
"""

gallery = ["bike", "bigbike", "moon", "cat", "moonshot", "moonlanding"]


def lucky() -> Any:
    g = np.random.choice(gallery)
    print("You got " + g + "!!!")
    return getattr(sys.modules["tensorcircuit"], g)


def set_ascii(b: str = "", conf: Optional[Dict[str, str]] = None) -> None:
    global visible
    md5 = hashlib.md5()
    b += "some pepper!!"
    md5.update(b.encode("utf-8"))
    if md5.hexdigest() in __passwd:
        print("Have fun!")
        visible = True
        if conf is None:
            conf = {}
        for g in gallery:
            s = getattr(thismodule, "__" + g + "__")
            # if s.find("%") > 0: # avoid % in the art
            if g in ["moonshot"]:
                s = s % conf.get(g, "")
            setattr(
                sys.modules["tensorcircuit"],
                g,
                Art(s),
            )
        setattr(
            sys.modules["tensorcircuit"],
            "lucky",
            lucky,
        )
    else:
        visible = False
        raise AttributeError("module 'tensorcircuit' has no attribute 'set_ascii'")


def __get_url(key: str) -> str:
    if visible:
        from Cryptodome.Cipher import AES

        def pad_key(key: str) -> str:
            while len(key) % 16 != 0:
                key += "1"
            return key

        aes = AES.new(pad_key(key).encode(), AES.MODE_ECB)
        #
        e = b"\xd8\xbcvw6G\xc4\xd7\xc3\x0f\xa7\xe4v\n\xcfcq-\x8b2\x90\xb5\x91\xbe\x8b\x7f\x96\x1df\xe1\xb1\x07,>\x8c\x99?\xa5\xeb\xc5se\xf0\xe0\xb3\xd5\x99<\xfa\xb2\x9c\xf6\x87\xc0el6\x88A\xb3\xeb\xfc\xe7AE\xe7\xca\xfc\xaf\x19!H\xb8\x8b\xba\xcd\xf7\xe7g\x83(U\xce\xf8n,\xfc\xd9\xa6\xe2\xf0#\x06\xe4\x0e\x04\xb7\x922\x18\x92(\x07O\x95p\xde\x95\xbb\x01\xeaZC\xa6\x8a\xc0\xa5\x11\x93\x00\xfe\xde\xdc\xba'\xb36\xef\xfa\x0fh\x9f\xf6\xa0\x02b\xe1:\xcb\x16\xec\xcc\xad\x94\xb1b\x92\xd3x\x16\xc4\x0cO%r\\\x9d\x11H\x13"  # pylint: disable = line-too-long
        #
        de = str(aes.decrypt(e), encoding="utf-8", errors="ignore")
        return de.strip()[10:-10]
    #
    #

    raise AttributeError("module 'tensorcircuit' has no attribute '__get_url'")


def __encrypt(key: str, url: str) -> Any:
    if visible:
        from Cryptodome.Cipher import AES

        def pad(text: str) -> str:
            while len(text) % 16 != 0:
                text += " "
            return text

        def pad_key(key: str) -> str:
            while len(key) % 16 != 0:
                key += "1"
            return key

        url = "salty!!!!!" + url + ">>>>><<<<<"

        aes = AES.new(pad_key(key).encode(), AES.MODE_ECB)

        encrypted_text = aes.encrypt(pad(url).encode())

        return encrypted_text


def get_message(key: str) -> str:
    if visible:
        from requests import get, RequestException

        url = __get_url(key)

        try:
            r = get(url)
        except RequestException:
            print("Bro, what R U doin?")
            return ""
        return r.text

    #
    #

    raise AttributeError("module 'tensorcircuit' has no attribute 'get_message'")
