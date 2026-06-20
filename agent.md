# Agent Notes

## Project Overview

This folder contains a MATLAB spike-sorting workflow built around Intan data files and Wave_Clus. The main user-facing scripts are:

- `spike_sorter.m`: interactive single-file workflow.
- `spike_sorter_group.m`: batch-style workflow over files in a selected folder.
- `spike_sorter_core.m`: core loader, filtering, spike detection, and return-package builder.

The package now supports both Intan RHD and RHS files.

## Intan File Loading

- `read_Intan_RHD2000_file_no_prompt_new.m` reads `.rhd` files.
- `read_Intan_RHS2000_file_no_prompt_new.m` reads `.rhs` files.
- Both readers return amplifier data as channels x samples:
  ```matlab
  [amplifier_data, frequency_parameters] = read_Intan_RHD2000_file_no_prompt_new(fname, pname);
  [amplifier_data, frequency_parameters] = read_Intan_RHS2000_file_no_prompt_new(fname, pname);
  ```
- `spike_sorter_core.m` routes by file extension and reads `sampleRate` from `frequency_parameters.amplifier_sample_rate`.
- Do not reintroduce a hardcoded `sampleRate=20000`; RHS files can use different sample rates.
- `spike_sorter_core.m` calls `cd(pname)`, so it also adds its own package folder to the MATLAB path before changing directories. Keep this behavior so sibling reader functions remain available.

## Important Data Shapes

- Reader output: rows are channels, columns are time samples.
- `spike_sorter_core.m` transposes selected data into samples x channels for filtering.
- Returned fields:
  - `returnPackage.unfilteredData`: samples x channels.
  - `returnPackage.filteredData`: samples x channels.
  - `returnPackage.sampleRate`: Hz, from Intan header.
  - `returnPackage.spikesGroup`: one cell per channel.
  - `returnPackage.indicesGroup`: spike sample indices per channel.

## MATLAB Validation Commands

MATLAB is available at:

```bash
/Applications/MATLAB_R2025a.app/bin/matlab
```

Direct RHS reader smoke test:

```bash
/Applications/MATLAB_R2025a.app/bin/matlab -batch "p='/Users/jialunz/SynDrive_IC/Jialun/Active mouse recording/20260617 surgery LP with IrOx/left_stimulation_anode_260618_104947/'; f='left_stimulation_anode_260618_104947.rhs'; [d, params]=read_Intan_RHS2000_file_no_prompt_new(f,p); fprintf('SIZE %d %d\n', size(d,1), size(d,2)); fprintf('SR %.12g\n', params.amplifier_sample_rate);"
```

Full sorter smoke test on the same RHS file:

```bash
/Applications/MATLAB_R2025a.app/bin/matlab -batch "p='/Users/jialunz/SynDrive_IC/Jialun/Active mouse recording/20260617 surgery LP with IrOx/left_stimulation_anode_260618_104947/'; f='left_stimulation_anode_260618_104947.rhs'; r=spike_sorter_core(f,p,0,0); fprintf('SR %.12g\n', r.sampleRate); fprintf('CHANNELS %d\n', r.numChannels); fprintf('UNFILTERED %d %d\n', size(r.unfilteredData,1), size(r.unfilteredData,2)); fprintf('FILTERED %d %d\n', size(r.filteredData,1), size(r.filteredData,2)); fprintf('SPIKE_GROUPS %d\n', numel(r.spikesGroup));"
```

Expected values for that RHS file:

- 29 amplifier channels.
- 30000 Hz sample rate.
- 1770240 samples for the full recording.
- `unfilteredData` and `filteredData` shape: `1770240 x 29`.

Static MATLAB check:

```bash
/Applications/MATLAB_R2025a.app/bin/matlab -batch "files={'read_Intan_RHS2000_file_no_prompt_new.m','read_Intan_RHD2000_file_no_prompt_new.m','spike_sorter_core.m'}; for k=1:numel(files), msgs=checkcode(files{k}); fprintf('CHECKCODE %s %d\n', files{k}, numel(msgs)); end"
```

`checkcode` currently reports warnings only, mostly pre-existing style warnings in `spike_sorter_core.m` and harmless reader output-initialization warnings.

## Editing Guidance

- Prefer small, local MATLAB changes; much of this code is research workflow code with many historical channel-reordering blocks.
- Preserve old one-output reader calls when changing readers.
- Keep RHS stimulation metadata parsing in the RHS reader, but the spike sorter currently consumes only amplifier data and sample-rate metadata.
- Avoid changing channel-reordering blocks unless the user explicitly asks for a specific probe or experiment layout.
