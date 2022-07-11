"""
Visualization on circuits
"""
import os
import subprocess
from uuid import uuid4
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def gate_name_trans(gate_name: str) -> Tuple[int, str]:
    r"""
    Translating from the gate name to gate information including the
    number of control qubits and the reduced gate name.

    :Example:

    >>> string = r'ccnot'
    >>> tc.vis.gate_name_trans(string)
    2 'not'

    :param gate_name: String of gate name
    :type gate_name: str
    :return: # of control qubits, reduced gate name
    :rtype: Tuple[int, str]
    """
    ctrl_number = 0
    while gate_name[ctrl_number] == "c":
        ctrl_number += 1
    return ctrl_number, gate_name[ctrl_number:]


def qir2tex(
    qir: List[Dict[str, Any]],
    n: int,
    init: Optional[List[str]] = None,
    measure: Optional[List[str]] = None,
    rcompress: bool = False,
    lcompress: bool = False,
    standalone: bool = False,
    return_string_table: bool = False,
) -> Union[str, Tuple[str, List[List[str]]]]:
    r"""
    Generate Tex code from 'qir' string to illustrate the circuit structure.
    This visualization is based on quantikz package.

    :Example:

    >>> qir=[{'index': [0], 'name': 'h'}, {'index': [1], 'name': 'phase'}]
    >>> tc.vis.qir2tex(qir,2)
    '\\begin{quantikz}\n\ ... \n\\end{quantikz}'

    :param qir: The quantum intermediate representation of a circuit in tensorcircuit.
    :type qir: List[Dict[str, Any]]
    :param n: # of qubits
    :type n: int
    :param init: Initial state, default is an all zero state '000...000'.
    :type init: Optional[List[str]]
    :param measure: Measurement Basis, default is None which means no
        measurement in the end of the circuit.
    :type measure: Optional[List[str]]
    :param rcompress: If true, a right compression of the circuit will be conducted.
        A right compression means we will try to shift gates from right to left if possible.
    Default is false.
    :type rcompress: bool
    :param lcompress: If true, a left compression of the circuit will be conducted.
        A left compression means we will try to shift gates from left to right if possible.
        Default is false.
    :type lcompress: bool
    :param standalone: If true, the tex code will be designed to generate a standalone document.
        Default is false which means the generated tex code is just a quantikz code block.
    :type standalone: bool
    :param return_string_table: If true, a string table of tex code will also be returned.
        Default is false.
    :type return_string_table: bool
    :return: Tex code of circuit visualization based on quantikz package. If return_string_table
        is true, a string table of tex code will also be returned.
    :rtype: Union[str, Tuple[str, List[List[str]]]]
    """
    # flag for applied layers
    flag = np.zeros(n, dtype=int)
    tex_string_table: List[List[str]] = [[] for _ in range(n)]

    # initial state presentation
    if init is None:
        for i in range(n):
            tex_string_table[i].append(r"\lstick{$\ket{0}$}")
    else:
        for i in range(n):
            if init[i]:
                tex_string_table[i].append(r"\lstick{$\ket{" + init[i] + r"}$}")
            else:
                tex_string_table[i].append(r"\lstick{}")

    # apply gates in qir
    for x in qir:

        idx = x["index"]
        gate_length = len(idx)
        if x["name"].startswith("invisible"):
            p = max(flag[min(idx) : max(idx) + 1]) + 1
            for i in range(min(idx), max(idx) + 1):
                tex_string_table[i] += [r"\qw "] * (p - flag[i] - 1)
                tex_string_table[i] += [r"\ghost{" + x["name"][7:] + "}\qw "]
                flag[i] = p
        else:
            ctrl_number, gate_name = gate_name_trans(x["name"])

            if x.get("ctrl", None):
                ctrlbits = x["ctrl"]
            else:
                ctrlbits = [1] * ctrl_number

            low_idx = min(idx[ctrl_number:])
            high_idx = max(idx[ctrl_number:])

            p = max(flag[min(idx) : max(idx) + 1]) + 1
            for i in range(min(idx), max(idx) + 1):
                tex_string_table[i] += [r"\qw "] * (p - flag[i])
                flag[i] = p

            # control qubits
            for i in range(ctrl_number):
                if ctrlbits[i]:
                    ctrlstr = r"\ctrl{"
                else:
                    ctrlstr = r"\octrl{"
                ctrli = idx[i]
                if ctrli < low_idx:
                    tex_string_table[ctrli][-1] = ctrlstr + str(low_idx - ctrli) + r"} "
                elif ctrli > high_idx:
                    tex_string_table[ctrli][-1] = (
                        ctrlstr + str(high_idx - ctrli) + r"} "
                    )
                else:
                    tex_string_table[ctrli][-1] = ctrlstr + r"} "

            # controlled gate
            for i in range(min(idx), max(idx) + 1):
                # r" \qw " rather than r"\qw " represent that a vline will cross at this point
                # (flag for further compression operation)
                if tex_string_table[i][-1] == r"\qw ":
                    tex_string_table[i][-1] = r" \qw "

            if gate_length - ctrl_number == 1:
                if gate_name == "not":
                    tex_string_table[idx[ctrl_number]][-1] = r"\targ{} "
                elif gate_name == "phase":
                    tex_string_table[idx[ctrl_number]][-1] = r"\phase{} "
                #             elif gate_name == "none":
                #                 tex_string_table[idx[ctrl_number]][-1] = r"\ghost{}\qw "
                else:
                    tex_string_table[idx[ctrl_number]][-1] = (
                        r"\gate{" + gate_name + r"} "
                    )
            else:
                # multiqubits gate case
                idxp = np.sort(idx[ctrl_number:])
                p = 0
                vl = 0
                hi = 0
                while p < len(idxp):
                    if vl != 0:
                        tex_string_table[idxp[p - 1]][-1] = (
                            tex_string_table[idxp[p - 1]][-1]
                            + r"\vcw{"
                            + str(vl)
                            + r"} "
                            + r"\vqw{"
                            + str(vl)
                            + r"} "
                        )
                    li = idxp[p]
                    while p < len(idxp) - 1:
                        if idxp[p + 1] - idxp[p] == 1:
                            p = p + 1
                        else:
                            break
                    hi = idxp[p]
                    tex_string_table[li][-1] = (
                        r"\gate[" + str(hi + 1 - li) + r"]{" + gate_name + r"} "
                    )
                    p = p + 1
                    if p < len(idxp):
                        vl = idxp[p] - idxp[p - 1]
                # delete qwires on gate's qubits
                for i in idx:
                    if tex_string_table[i][-1] == r" \qw ":
                        tex_string_table[i][-1] = " "

    p = max(flag)
    for i in range(n):
        tex_string_table[i] += [r"\qw "] * (p - flag[i])

    #             # old version: linethrought
    #             for i in range(low_idx, high_idx + 1):
    #                 if (tex_string_table[i][-1] == r"\qw "):
    #                     tex_string_table[i][-1] = r"\linethrough "
    #             for i in idx[ctrl_number:]:
    #                 tex_string_table[i][-1] = r" "
    #             tex_string_table[low_idx][-1] = r"\gate[" + str(high_idx + 1 - low_idx) + r"]{" + gate_name + r"} "

    # right compression
    if rcompress:
        for i in range(n):
            while (tex_string_table[i][-1] == r"\qw ") | (
                tex_string_table[i][-1] == r" \qw "
            ):
                if tex_string_table[i][-1] == r"\qw ":
                    tex_string_table[i].pop()
                else:
                    p = 1
                    while p < len(tex_string_table[i]):
                        if tex_string_table[i][-p] == r" \qw ":
                            p += 1
                        else:
                            break
                    if tex_string_table[i][-p] == r"\qw ":
                        tex_string_table[i] = tex_string_table[i][:-p]
                    else:
                        break
    # left compression
    if lcompress:
        for i in range(n):
            p = 0
            lstring = len(tex_string_table[i])
            while p + 1 < lstring - 1:
                if tex_string_table[i][p + 1] == r"\qw ":
                    tex_string_table[i][p + 1] = r" "
                    p += 1
                else:
                    break
            tmp = tex_string_table[i][0]
            tex_string_table[i][0] = tex_string_table[i][p]
            tex_string_table[i][p] = tmp

    # measurement
    if measure is None:
        for i in range(n - 1):
            tex_string_table[i].append(r"\qw \\")
        tex_string_table[n - 1].append(r"\qw ")
    else:
        for i in range(n):
            if not measure[i]:
                tex_string_table[i].append(r"\qw \\")
            else:
                tex_string_table[i].append(r"\meter{" + measure[i] + r"} \\")
        tex_string_table[-1][-1] = tex_string_table[-1][-1][:-2]

    texcode = r"\begin{quantikz}" + "\n"
    for i in range(n):
        for x in tex_string_table[i]:  # type: ignore
            texcode += x + r"&"  # type: ignore
        texcode = texcode[:-1] + "\n"
    texcode += r"\end{quantikz}"
    if standalone:
        texcode = (
            r"""\documentclass{standalone}
\usepackage{quantikz}
\begin{document}
"""
            + texcode
            + r"""
\end{document}"""
        )
    if return_string_table:
        return texcode, tex_string_table
    else:
        return texcode


def render_pdf(
    texcode: str,
    filename: Optional[str] = None,
    latex: Optional[str] = None,
    filepath: Optional[str] = None,
    notebook: bool = False,
) -> Any:
    r"""
    Generate the PDF file with given latex string and filename.
    Latex command and file path can be specified.
    When notebook is True, convert the output PDF file to image and return a Image object.

    :Example:

    >>> string = r'''\documentclass[a4paper,12pt]{article}
    ... \begin{document}
    ... \title{Hello TensorCircuit!}
    ... \end{document}'''
    >>> tc.vis.render_pdf(string, "test.pdf", notebook=False)
    >>> os.listdir()
    ['test.aux', 'test.log', 'test.pdf', 'test.tex']

    :param texcode: String of latex content
    :type texcode: str
    :param filename: File name, defaults to random UUID `str(uuid4())`
    :type filename: Optional[str], optional
    :param latex: Executable Latex command, defaults to `pdflatex`
    :type latex: Optional[str], optional
    :param filepath: File path, defaults to current working place `os.getcwd()`
    :type filepath: Optional[str], optional
    :param notebook: [description], defaults to False
    :type notebook: bool, optional
    :return: if notebook is True, return `Image` object; otherwise return `None`
    :rtype: Optional[Image], defaults to None
    """

    if not filepath:
        filepath = os.getcwd()
    if not latex:
        latex = "pdflatex"
    if not filename:
        filename = str(uuid4())
    if filename.endswith(".pdf"):
        filename = filename[:-4]
    texfile = os.path.join(filepath, filename + ".tex")
    with open(texfile, "w") as f:
        f.write(texcode)
    _ = subprocess.run(
        [
            latex,
            "-output-directory",
            filepath,
            texfile,
        ],
        stdout=subprocess.DEVNULL,
    )
    if notebook:
        # from IPython.display import IFrame

        # assert width is not None, ValueError("width must be a number")
        # assert height is not None, ValueError("width must be a number")
        # return IFrame(filename + ".pdf", width=width, height=height)
        from wand.image import Image

        return Image(filename=filename + ".pdf", resolution=300)
