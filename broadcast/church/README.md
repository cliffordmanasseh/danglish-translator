# Ferrum audio previews

These are **preview WAVs** made from the raw float32 captures. Audio is
48 kHz, mono, 16-bit PCM. Each file is peak-normalized to about -2 dBFS so it
is audible; the raw captures are untouched.

Important: files 01–06 are **legacy** captures from before the C11 core.
Files 07–10 are current C11 Osc1 captures. There is no current C11 release
tail or re-strike audio file.

```text
01 stock note36                 legacy stock oscillator
02 Ferrum old note36            legacy Ferrum kernel
03 Ferrum old filter low        legacy Ferrum + filter
04 Ferrum old filter high       legacy Ferrum + filter
05 official patch filter low    legacy official-patch capture
06 official patch filter high   legacy official-patch capture
07 C11 A=0                      current C11, A low
08 C11 A=127                    current C11, A high
09 C11 B=0                      current C11, B low
10 C11 B=127                    current C11, B high
```

Raw sources live in `out/ferrum_g0/panel_fix/`:

```text
pitch_clean_stock.bin
pitch_clean.bin
filter_low.bin
filter_high.bin
patch_filter_low.bin
patch_filter_high.bin
c11_a0.pcm ... c11_b127.pcm
```

The previews are normalized, so do not use them to compare absolute levels.
No listening judgment was made by the agent.

## Looped previews

Files ending `_loop4.wav` repeat the short capture four times with 10 ms fades.
Use these for quick listening. The original short files are kept beside them.

```text
07_c11_a0_loop4.wav       1.92 s
08_c11_a127_loop4.wav     1.92 s
09_c11_b0_loop4.wav       1.92 s
10_c11_b127_loop4.wav     1.92 s
```

Easy path:

```text
/home/clmm/ferrum_audio
```

## Six modal algorithm previews

Current C11 Osc1, default A/B, one note:

```text
11_c11_algo0_hammered_bar.wav
12_c11_algo1_marimba.wav
13_c11_algo2_bell.wav
14_c11_algo3_glass_bell.wav
15_c11_algo4_wood_block.wav
16_c11_algo5_steel_tongue.wav
```

Each has a `_loop4.wav` version. Stock table entries 0..5 now point to the
Ferrum dispatcher; entries 6..39 remain stock.

## String family previews

Current C11 Osc1, default A/B, one note:

```text
17_c11_algo6_nylon_pluck.wav
18_c11_algo7_steel_string.wav
19_c11_algo8_harp.wav
20_c11_algo9_koto.wav
```

Each has a `_loop4.wav` version. Stock table entries 0..9 now point to the
Ferrum dispatcher; entries 10..39 remain stock.
