#!/usr/bin/env python3

import os
import utils
from bench_conf import params

def incr(arr, lims):
    if arr[0] == lims[0]-1:
        if len(arr) == 1:
            return None
        i = incr(arr[1:], lims[1:])
        if i is None:
            return None
        arr[1:] = i
        arr[0] = 0
    else:
        arr[0] += 1
    return arr


def get_perm(d):
    keys = d.keys()
    values = d.values()
    var_num = len(keys)
    p = [0] * var_num
    plen = list(map(len, values))
    while p is not None:
        sel = {}
        for k,v,pn in zip(keys, values,p):
            sel[k] = v[pn]
        yield sel
        p = incr(p, plen)

def build_cmd(params):
    l = []
    for k in params:
        l.append("--{}={}".format(k, params[k]))
    return " ".join(l)

def run(cmd):
    if utils.find_argv("dry", False):
        print("Dry run:{}".format(cmd))
    else:
        os.system(cmd)

def main():
    for d in get_perm(params):
        cmd = "./classify.py --method=nn {}".format(build_cmd(d))
        run(cmd)

if __name__ == "__main__":
    main()
