#!/bin/sh

outdir="benchmarks/$(date -Iseconds)/"

mkdir -p $outdir

cat /proc/cpuinfo >> $outdir/cpuinfo
clinfo >> $outdir/clinfo
./bench.py >> $outdir/results


