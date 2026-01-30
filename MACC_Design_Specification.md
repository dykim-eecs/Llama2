# MACC (Memory Access Controller with CRC) Design Specification

## 1. General Description

**MACC** is a hardware module that provides CRC-based error detection and correction capabilities to ensure SRAM data reliability. This module performs the following core functions:

- **Write Path:** Generates error detection codes through CRC encoding during data storage.
- **Read Path:** Detects and corrects errors in real-time during data retrieval.

MACC is categorized into three types based on the connected SRAM's port configuration and clock domains:

| **Item** | **Single Port (SP)** | **Two Port (TP)** | **Dual Port (DP)** |
|----------|---------------------|-------------------|-------------------|
| **Clock Output** | mem0_clk | mem0_clk | mem0_clk_a, mem0_clk_b |
| **ICG Count** | 1 | 1 | 2 |
| **mac_config Width** | `[4:0]` (5-bit) | `[4:0]` (5-bit) | `[7:0]` (8-bit) |
| **cfg_mcyc** | Shared | Shared | Separate (rd/wr) |
| **Port Usage** | Time-multiplexed R/W | Port A: Write, Port B: Read | Independent R/W |
| **Arbitration** | Round-Robin | None (dedicated ports) | None (dedicated ports) |

### 1.1 Key Features

- **Iterative Decoding (CID):** When corrupted data is detected, attempts recovery through up to 2-bit Correctable Iterative Decoding.
- **Partial SRAM Initialization:** Enables fast initialization of specific address ranges during system startup or on-demand, improving efficiency.
- **Error Injection:** Provides intentional error injection capability to verify that the system's error handling flow operates correctly.

### 1.2 Theory of Operation

MACC operates between the Master IP and SRAM. All request ordering guarantees and Hazard Detection (in DP mode) are assumed to be managed by the Master IP. MACC focuses on SRAM access efficiency and data integrity.

**Key Assumptions:**
- Master IP handles all read-after-write (RAW) and write-after-read (WAR) hazards
- Request ordering is maintained by the upstream controller
- MACC provides best-effort delivery with CRC protection

---

## 2. Architecture Design

### 2.1 Top Module

| Module | Function | Key Characteristics |
|--------|----------|---------------------|
| **AGMW** | Write Path | CRC encoding, SRAM initialization, response generation |
| **AGMR** | Read Path | CRC verification, 1/2-bit error correction, bypass path |
| **AGMS** | Memory Access | Round-Robin arbitration (SP), independent ports (TP/DP) |
| **TOP** | Integration | mac_config parsing, submodule interconnection |

### 2.2 Interface

#### 2.2.1 MAC Global Interface

| Signal | Width | Direction | Description |
|--------|-------|-----------|-------------|
| `mac_clk` | 1 | Input | Main clock |
| `mac_arstn` | 1 | Input | Asynchronous reset (active-LOW) |
| `mac_config` | 5 or 8 | Input | Configuration register |
| `mac_trigger` | 3+2×W_ADDR | Input | Trigger signals (clear, err_inj, init) |
| `mac_status` | 2+W_CRC | Output | Status register (init_busy, cid_busy, poly) |

**mac_config Fields (SP/TP - 5-bit):**

| Bits | Field | Description |
|------|-------|-------------|
| [2:0] | cfg_mcyc | Memory cycle configuration (1~7) |
| [4:3] | crc_bit | CID correction mode (00=disable, 01=1-bit, 10=2-bit) |

**mac_config Fields (DP - 8-bit):**

| Bits | Field | Description |
|------|-------|-------------|
| [2:0] | cfg_mcyc_wr | Write memory cycle (1~7) |
| [5:3] | cfg_mcyc_rd | Read memory cycle (1~7) |
| [7:6] | crc_bit | CID correction mode |

**mac_trigger Fields:**

| Bits | Field | Description |
|------|-------|-------------|
| [0] | clear_start | Clear signal (level trigger) |
| [1] | err_inj_en | Error injection enable |
| [2] | init_start | Initialization start (level trigger) |
| [3+W_ADDR-1:3] | init_start_addr | Init start address |
| [3+2×W_ADDR-1:3+W_ADDR] | init_end_addr | Init end address (0=full range) |

#### 2.2.2 MAC Handshake Interface (MHS)

**Write Request Interface:**

| Signal | Width | Direction | Description |
|--------|-------|-----------|-------------|
| `mhs_wreq_valid` | 1 | Input | Write request valid |
| `mhs_wreq_addr` | W_ADDR | Input | Write address |
| `mhs_wreq_data` | W_DATA | Input | Write data |
| `mhs_wreq_meta` | W_META | Input | Metadata (tag/ID) |
| `mhs_wreq_ready` | 1 | Output | Write request ready |

**Write Response Interface:**

| Signal | Width | Direction | Description |
|--------|-------|-----------|-------------|
| `mhs_wrsp_valid` | 1 | Output | Write response valid |
| `mhs_wrsp_meta` | W_META | Output | Response metadata |
| `mhs_wrsp_ready` | 1 | Input | Write response ready |

**Read Request Interface:**

| Signal | Width | Direction | Description |
|--------|-------|-----------|-------------|
| `mhs_rreq_valid` | 1 | Input | Read request valid |
| `mhs_rreq_addr` | W_ADDR | Input | Read address |
| `mhs_rreq_meta` | W_META | Input | Metadata (tag/ID) |
| `mhs_rreq_ready` | 1 | Output | Read request ready |

**Read Response Interface:**

| Signal | Width | Direction | Description |
|--------|-------|-----------|-------------|
| `mhs_rrsp_valid` | 1 | Output | Read response valid |
| `mhs_rrsp_data` | W_DATA | Output | Read data |
| `mhs_rrsp_meta` | W_META | Output | Response metadata |
| `mhs_rrsp_erpt` | Variable | Output | Error report |
| `mhs_rrsp_ready` | 1 | Input | Read response ready |

#### 2.2.3 MAC General Memory Interface (MGM)

**Internal Interface (AGMW/AGMR ↔ AGMS):**

| Signal | Width | Direction | Description |
|--------|-------|-----------|-------------|
| `mgm_wvalid` | 1 | AGMW→AGMS | Write request valid |
| `mgm_waddr` | W_ADDR | AGMW→AGMS | Write address |
| `mgm_wcode` | W_CODE | AGMW→AGMS | Encoded data (data + CRC) |
| `mgm_wget` | 1 | AGMS→AGMW | Write request accepted |
| `mgm_rvalid` | 1 | AGMR→AGMS | Read request valid |
| `mgm_raddr` | W_ADDR | AGMR→AGMS | Read address |
| `mgm_rerr_inj` | 1 | AGMR→AGMS | Error injection flag |
| `mgm_rget` | 1 | AGMS→AGMR | Read request accepted |
| `mgm_rret` | 1 | AGMS→AGMR | Read data return valid |
| `mgm_rcode` | W_CODE | AGMS→AGMR | Returned encoded data |

### 2.3 Micro-Architecture

#### (1) Single Port Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │                  MACC TOP                    │
                    │                                             │
  MHS Write ──────► │  ┌─────────┐                               │
                    │  │  AGMW   │──► MGM Write ──┐              │
                    │  │         │                │              │
                    │  └─────────┘                ▼              │
                    │                        ┌─────────┐          │
                    │                        │  AGMS   │◄──► SRAM │
                    │                        │  (RR)   │          │
                    │                        └─────────┘          │
                    │  ┌─────────┐                ▲              │
  MHS Read  ──────► │  │  AGMR   │──► MGM Read ───┘              │
                    │  │         │◄── MGM Return                 │
                    │  └─────────┘                               │
                    └─────────────────────────────────────────────┘
```

**Key Characteristics:**
- Single SRAM port shared between read and write
- Round-Robin arbiter in AGMS alternates between write and read requests
- Memory cycle (mcyc) defines the clock gating period

#### (2) Two Port / Dual Port Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │                  MACC TOP                    │
                    │                                             │
  MHS Write ──────► │  ┌─────────┐                               │
                    │  │  AGMW   │──► Port A (Write) ──► SRAM    │
                    │  └─────────┘                      Port A   │
                    │                                             │
                    │  ┌─────────┐                               │
  MHS Read  ──────► │  │  AGMR   │◄─► Port B (Read)  ◄─► SRAM    │
                    │  └─────────┘                      Port B   │
                    └─────────────────────────────────────────────┘
```

**Key Characteristics:**
- Dedicated ports eliminate arbitration overhead
- TP: Shared clock for both ports
- DP: Independent clocks allowing different frequencies

---

## 3. Parameters

Parameters are divided into Fixed Parameters (set at generation time) and Runtime Configuration (changeable during operation).

### 3.1 Fixed Parameters

| Parameter | Description | Typical Value | Notes |
|-----------|-------------|---------------|-------|
| `W_DATA` | Data width | 256 | User data bits |
| `W_CRC` | CRC width | 16 | CRC polynomial degree |
| `W_CODE` | Codeword width | W_DATA + W_CRC | Total stored bits |
| `W_META` | Metadata width | 8 | Tag/ID for tracking |
| `W_ADDR` | Address width | 12 | 4096 entries |
| `POLY` | CRC polynomial | 0x8005 | CRC-16-IBM |
| `FIFO_DEPTH` | AGMW FIFO depth | 4 | Power of 2 |
| `DATA_FIFO_DEPTH` | AGMR FIFO depth | 2 | ostd/code FIFO |
| `ERR_INJ_NUM` | Error injection bits | 2 | Number of bit positions |
| `ERR_INJ_POS` | Injection positions | {0, 1} | Bit indices to flip |

### 3.2 Runtime Configuration

| Field | Width | Description | Valid Range |
|-------|-------|-------------|-------------|
| `cfg_mcyc` | 3 | Memory cycle count | 1~7 |
| `crc_bit` | 2 | CID mode | 00=off, 01=1-bit, 10=2-bit |
| `err_inj_en` | 1 | Error injection enable | 0/1 |
| `init_start` | 1 | Start initialization | Rising edge trigger |
| `init_start_addr` | W_ADDR | Init start address | 0 ~ 2^W_ADDR-1 |
| `init_end_addr` | W_ADDR | Init end address | 0=full range |

---

## 4. Operating Modes

Operating modes are categorized into: (1) Normal Write, (2) Normal Read, (3) Initialization, and (4) CRC Iterative Decoding.

### 4.1 Normal Write

**Data Flow:**
```
MHS Request → SBUF (2-entry) → CRC Encoder → FIFO → AGMS → SRAM
                                                  ↓
                                           Response Controller → MHS Response
```

**Timing Diagram:**
```
           ┌───┐   ┌───┐   ┌───┐   ┌───┐
mac_clk    │   │   │   │   │   │   │   │
         ──┘   └───┘   └───┘   └───┘   └──

         ──────┐                   ┌──────
wreq_valid     │                   │
               └───────────────────┘

         ──────────────┐       ┌──────────
wreq_ready             │       │
                       └───────┘

                           ┌───────┐
wrsp_valid                 │       │
         ──────────────────┘       └──────
```

**Key Points:**
- Ready deasserts when SBUF or FIFO is full
- Response is generated after SRAM write is accepted
- Zero-latency response support available

### 4.2 Normal Read

**Data Flow:**
```
MHS Request → SBUF (optional) → Outstanding FIFO ──┐
                                                   ↓
                                     AGMS Request (mgm_rvalid)
                                                   ↓
                                     SRAM Read → Code FIFO
                                                   ↓
                              Syndrome Calculator (CRC check)
                                      ↓                    ↓
                            syndrome ≠ 0             syndrome = 0
                                      ↓                    ↓
                                  Corrector          Bypass Path
                                      ↓                    ↓
                              Response Generator ──────────┘
                                      ↓
                                MHS Response + Error Report
```

**Error Report Format (mhs_rrsp_erpt):**

| Bits | Field | Description |
|------|-------|-------------|
| [0] | UE | Uncorrectable Error |
| [1] | CE | Correctable Error |
| [6:2] | Ecnt | Error count (5-bit) |
| [6+W_EPOS:7] | Epos1 | First error position |
| [6+2×W_EPOS:7+W_EPOS] | Epos2 | Second error position |
| [...] | Eaddr | Error address |
| [...] | Synd | Syndrome value |
| [MSB:...] | Odata | Original (corrupted) data |

### 4.3 Initialization

**State Machine:**
```
    ┌──────┐
    │ IDLE │◄───────────────────────────────────┐
    └──┬───┘                                    │
       │ init_start                             │
       ▼                                        │
┌─────────────┐                                 │
│ WAIT_DRAIN  │ Wait for all FIFOs empty        │
└──────┬──────┘                                 │
       │ all_fifos_empty                        │
       ▼                                        │
  ┌─────────┐                                   │
  │ RUNNING │ Write zeros to address range      │
  └────┬────┘                                   │
       │ addr >= end_addr                       │
       ▼                                        │
  ┌──────────┐                                  │
  │ COOLDOWN │ Wait 16 cycles for pipeline      │
  └────┬─────┘                                  │
       │ cooldown_cnt >= 15                     │
       ▼                                        │
  ┌──────────┐                                  │
  │ COMPLETE │──────────────────────────────────┘
  └──────────┘
```

**Key Points:**
- `mhs_wreq_ready` and `mhs_rreq_ready` are deasserted immediately when `init_start` rises
- WAIT_DRAIN ensures all in-flight requests complete before initialization
- Writes zero-encoded data to the specified address range
- Cooldown period ensures all pipeline stages are flushed

### 4.4 CRC Iterative Decoding (CID)

**Algorithm:**
1. Calculate syndrome: `syndrome = CRC(received_data) ⊕ received_CRC`
2. If syndrome = 0: No error, bypass to output
3. If syndrome ≠ 0 and crc_bit = 00: Report UE (correction disabled)
4. If syndrome ≠ 0 and crc_bit = 01: Attempt 1-bit correction
5. If syndrome ≠ 0 and crc_bit = 10: Attempt 2-bit correction

**1-bit Correction:**
```
for each bit position i in [0, W_CODE-1]:
    if syndrome == column_i of generator_matrix:
        flip bit i → corrected data
        return CE
return UE
```

**2-bit Correction:**
```
for each pair (i, j) where i < j in [0, W_CODE-1]:
    if syndrome == (column_i ⊕ column_j):
        flip bits i and j → corrected data
        return CE
return UE
```

**Performance Impact:**
- 1-bit mode: O(W_CODE) iterations worst case
- 2-bit mode: O(W_CODE²) iterations worst case
- `cid_busy` signal indicates correction in progress

---

## 5. SDC (Timing Constraints)

**Setup Margin:** 15% (SCALE_FACTOR = 0.85) applied

**Clock Definitions:**
```tcl
# Main clock
create_clock -name mac_clk -period $CLK_PERIOD [get_ports mac_clk]

# For Dual Port: separate clock domains
create_clock -name mac_clk_a -period $CLK_PERIOD_A [get_ports mac_clk]
create_clock -name mac_clk_b -period $CLK_PERIOD_B [get_ports mac_clk]
```

**False Paths:**
```tcl
# Asynchronous reset
set_false_path -from [get_ports mac_arstn]

# Configuration registers (quasi-static)
set_false_path -from [get_ports mac_config*]
```

**Memory Interface Constraints:**
```tcl
# Output delay to SRAM
set_output_delay -clock mac_clk -max [expr $CLK_PERIOD * 0.3] [get_ports mem0_*]
set_output_delay -clock mac_clk -min 0.1 [get_ports mem0_*]

# Input delay from SRAM
set_input_delay -clock mac_clk -max [expr $CLK_PERIOD * 0.3] [get_ports mem0_dout*]
set_input_delay -clock mac_clk -min 0.1 [get_ports mem0_dout*]
```

---

## 6. Testbenches

Four testbenches are provided for MACC module verification:

### TB1: tb_mcyc_test

**Purpose:** Verify memory access timing across various mcyc values

**Test Scenario:**
1. **Write Phase:** Consecutive writes of 10 entries each for mcyc = 2,3,4,5,6,7,1
   - Addresses: 0x100 (mcyc=2), 0x200 (mcyc=3), ..., 0x700 (mcyc=1)
   - 10-cycle wait between each mcyc group
2. **Read Phase:** Consecutive reads in the same order (10 entries each)
3. **Verification:** Compare read data with written data

**Pass Criteria:** All 70 read data entries match corresponding write data

**Coverage:**
- Memory cycle timing verification
- Back-to-back request handling
- mcyc runtime reconfiguration

---

### TB2: tb_crc_error_report

**Purpose:** Verify CRC error detection/correction and error report generation

**Test Scenario:**
1. **Write:** Single data entry at address 0x100 (mcyc=2)
2. **Read 4 times** (same address with error injection):

| Read# | mcyc | crc_bit | Expected | Description |
|-------|------|---------|----------|-------------|
| 0 | 2 | 00 | UE | Correction disabled |
| 1 | 3 | 01 | UE | 1-bit mode cannot correct 2-bit error |
| 2 | 4 | 10 | CE | 2-bit mode corrects successfully |
| 3 | 1 | 00 | UE | Correction disabled |

**Pass Criteria:** Each crc_bit mode produces expected UE/CE result

**Coverage:**
- Error injection mechanism
- Syndrome calculation
- 1-bit and 2-bit correction paths
- Error report format verification

---

### TB3: tb_concurrent_load

**Purpose:** Verify Round-Robin Arbiter operation under simultaneous Write/Read requests

**Test Scenario:**
1. **Pre-write:** Write data to addresses 0x200~0x209 for subsequent reads
2. **Concurrent Requests:** Fork-join simultaneous Write and Read
   - Write: Addresses 0x100~0x109, 10 consecutive entries
   - Read: Addresses 0x200~0x209, 10 consecutive entries
3. **Response Verification:** Confirm all 10 responses received for each path

**Pass Criteria:**
- 10 write responses received
- 10 read responses received with correct data
- No deadlock detected

**Coverage:**
- Arbiter fairness
- Concurrent request handling
- Pipeline throughput under load

---

### TB4: tb_init_interrupt

**Purpose:** Verify request blocking during Init operation

**Test Scenario:**

**TEST 1: Init Interrupt During Write**
1. Start 10 consecutive writes (addresses 0x300~0x309, mcyc=2)
2. Trigger Init after 5th handshake
3. **Verify:** `mhs_wreq_ready = 0` during init
4. Confirm remaining 5 writes complete after Init

**TEST 2: Init Interrupt During Read**
1. Start 10 consecutive reads (addresses 0x300~0x309, mcyc=2)
2. Trigger Init after 5th handshake
3. **Verify:** `mhs_rreq_ready = 0` during init
4. Confirm remaining 5 reads complete after Init

**Pass Criteria:**
- Ready signals deassert immediately when init_start rises
- Requests blocked throughout init operation
- Normal operation resumes after init completion
- In-flight requests (before init) complete successfully

**Coverage:**
- Init blocking mechanism
- FIFO drain behavior
- State machine transitions
- Request pipeline integrity

---

## 7. Code Generation

MACC RTL is generated using the `generate_system.py` script with user-specified parameters.

**Usage:**
```bash
python3 scripts/generate_system.py <user_prefix> [--port single|tp|dp]
```

**Generated Files:**
```
code/<config_folder>/
├── rtl/macc/
│   ├── rbln_<prefix>_agm_wr.sv      # Write module
│   ├── rbln_<prefix>_agm_rd.sv      # Read module
│   ├── rbln_<prefix>_agm_m2s.sv     # Memory access module
│   ├── rbln_<prefix>_agm_top.sv     # Top module
│   └── module/                       # Dependencies
├── tb/                               # Testbenches
├── filelist/                         # File lists for simulation
└── sim/                              # Simulation scripts
```
