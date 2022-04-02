"""
Some ascii art from https://www.asciiart.eu/, have fun!
"""

import hashlib
import sys
from typing import Any, Dict, Optional

import numpy as np

visible = False
__passwd = {"7f935440a50f08f25354f5daa334282c"}
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
    md5.update(b.encode("utf-8"))
    if md5.hexdigest() in __passwd:
        print("Have fun!")
        visible = True
        if conf is None:
            conf = {}
        for g in gallery:
            s = getattr(thismodule, "__" + g + "__")
            # if s.find("%") > 0: # avoid % in the art
            if s in ["__moonshot__"]:
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
