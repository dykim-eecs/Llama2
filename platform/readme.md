# Vitis 2025.1: Fix for rlwrap‑related crash (Segmentation fault)

In Vitis 2025.1, you might see a **Segmentation fault** during the `xsct` / `xtclsh` / `cf2bd` stage. This has been reported as a crash caused by the **bundled `rlwrap`**. Follow the steps below to resolve it.

---

## 1) Replace the bundled rlwrap (most effective)

**Install the system `rlwrap`:**

```bash
sudo apt-get update
sudo apt-get install -y rlwrap
```

**Go to Vitis' unwrapped binary folder (adjust path to your install):**

```bash
cd /tools/Xilinx/2025.1/Vitis/bin/unwrapped/lnx64.o
```

**Back up the bundled rlwrap and symlink the system one:**

```bash
sudo mv rlwrap rlwrap.old
sudo ln -s /usr/bin/rlwrap rlwrap
```

Then run the same build again.
(This is the same workaround used on WSL/some Linux distros where `xsct`/`xtclsh`/`cf2bd` segfault due to `rlwrap`.)

---

## 2) Ensure `tcl` is available (usually already installed)

```bash
which tclsh8.6 || sudo apt-get install -y tcl8.6
```

---

## 3) Clean and retry

**Quick clean:**

```bash
rm -rf _x build .Xil
```

**Reload environment:**

```bash
source /tools/Xilinx/2025.1/Vitis/settings64.sh
source /opt/xilinx/xrt/setup.sh   # adjust XRT path to your system
```

**Rebuild:**

```bash
make
```

---

### Notes

* Installation paths (e.g., `/tools/Xilinx/2025.1/...`) may vary by system.
* Avoid building with `sudo make`—environment variables may differ and cause failures.
* If the issue persists, check for a 2025.1 hotfix or try a nearby release (e.g., 2024.2) to compare behavior.
