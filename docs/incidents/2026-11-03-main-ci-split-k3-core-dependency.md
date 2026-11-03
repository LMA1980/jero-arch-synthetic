# Incident Report: `LMA1980/k3-arch-synthetic` `main` CI Failure Due to Split `k3-core` Git Dependency Graph (`job-1793671352-9979101a`)

- **Date**: 2026-11-03
- **Status**: Resolved (Upstream fix PR opened on `LMA1980/k3-kernels-rvv#42`; CI v3 workflow & incident report opened on `LMA1980/k3-arch-synthetic`)
- **Affected Repository**: `LMA1980/k3-arch-synthetic` (`main` branch, commit `9979101a`)
- **Failed Job ID**: `job-1793671352-9979101a` (`clippy-check` stage on `k3-ci`)

---

## 1. Incident Summary & Impact

On 2026-11-03, the `main` branch CI pipeline for `LMA1980/k3-arch-synthetic` (`job-1793671352-9979101a`, commit `9979101a`) failed during the `clippy-check` (`cargo clippy`) stage with two `E0277` type-mismatch errors on `k3_core::K3Error` and `k3_core::TensorView`:

```text
error[E0277]: `?` couldn't convert the error to `k3_core::K3Error`
  --> src/lib.rs:41:82
   |
41 |                     BaselineKernel::new().vec_dot_q4_k_m(&mut q_proj, &q_tensor, &hidden)?;
   |                     -------------------------------------------------------------^^^^^^^^-
   |                     |                                                            |
   |                     |                                                            expected `&k3_core::tensor::TensorView<'_>`, found `&k3_core::TensorView<'_>`
   |                     this can't be annotated with `?` because it has type `Result<_, k3_core::error::K3Error>`
   |
   = note: `k3_core::error::K3Error` and `k3_core::K3Error` have similar names, but are actually distinct types
   = note: perhaps two different versions of crate `k3_core` are being used?
```

### Impact
- Standalone `main`-branch builds and `k3-ci` verification runs of `LMA1980/k3-arch-synthetic` (when built outside a parent Cargo workspace `[patch]` table) failed to compile whenever the default `with-kernels` (`dep:kernels-rvv`) feature was enabled.
- Downstream consumers pulling both `k3-core` and `kernels-rvv` from default `main` branches without explicit `branch = "..."` specifiers resolved two distinct instances of `k3-core` in their Cargo dependency graph.

---

## 2. Root Cause Analysis

1. **Direct Dependency in `k3-arch-synthetic` (`origin/main`)**:
   `k3-arch-synthetic/Cargo.toml` on `main` declares its dependencies against default repository branches (no `branch = "..."` parameter):
   ```toml
   [dependencies]
   k3-core = { version = "0.1.0", git = "https://github.com/LMA1980/k3-core.git" }
   kernels-rvv = { version = "0.1.0", git = "https://github.com/LMA1980/k3-kernels-rvv.git", optional = true }
   ```
2. **Transitive Dependency in `k3-kernels-rvv` (`origin/main`)**:
   When `LMA1980/k3-kernels-rvv` PR `#37` (`bfdef6b`, `feat(kernels): implement shared RoPE, GQA, SwiGLU & quantized kernels`) was merged into `main`, its `Cargo.toml` still contained a feature-branch specifier for `k3-core`:
   ```toml
   [dependencies]
   k3-core = { version = "0.1.0", git = "https://github.com/LMA1980/k3-core.git", branch = "feat/m4-epic1-llama-mistral-phi3" }
   ```
3. **Cargo Git Source Identity Rules**:
   Cargo canonicalizes git dependency sources using the tuple `(git_url, reference_kind)` where `reference_kind` distinguishes `DefaultBranch` (omitted `branch`) from `Branch("feat/m4-epic1-llama-mistral-phi3")` and `Branch("main")`. Even if two branches point to compatible or identical commits, Cargo treats `git = ".../k3-core.git"` (default branch) and `git = ".../k3-core.git", branch = "feat/..."` as **two completely separate crate sources**.
4. **Why Parent Workspace Builds Masked the Issue**:
   Inside the parent multi-crate workspace, root `Cargo.toml` applies `[patch."https://github.com/LMA1980/k3-core.git"]` pointing to the local `code/core` path, unifying `k3-core` across all workspace members. However, standalone submodule CI (`k3-ci` running in `LMA1980/k3-arch-synthetic`) resolves git dependencies directly from GitHub without the parent workspace patch table, exposing the split `k3-core` graph immediately.

---

## 3. Scope Audit Across `k3-*` Repositories (`origin/main`)

An audit of `origin/main:Cargo.toml` across all `k3-*` repositories revealed the following git dependency reference specifiers on `main`:

| Repository | `origin/main` Git Dependency Specifiers | Status on `main` |
| :--- | :--- | :--- |
| `LMA1980/k3-core` | None (leaf foundation crate) | Clean |
| `LMA1980/k3-tokenizer` | `k3-core` (default branch, no `branch = "..."`) | Clean |
| `LMA1980/k3-arch-synthetic` | `k3-core`, `kernels-rvv` (default branch, no `branch = "..."`) | Clean (victim of upstream `k3-kernels-rvv` pin) |
| `LMA1980/k3-kernels-rvv` | `k3-core` pinned to `branch = "feat/m4-epic1-llama-mistral-phi3"` | **Affected** (Fixed in `LMA1980/k3-kernels-rvv#42`) |
| `LMA1980/k3-arch-llama` | `k3-core`, `kernels-rvv` pinned to `branch = "feat/m4-epic1-llama-mistral-phi3"` | **Affected** (Retains feature-branch pin on `main`) |
| `LMA1980/k3-arch-mistral` | `k3-core`, `kernels-rvv` pinned to `branch = "feat/m4-epic1-llama-mistral-phi3"` | **Affected** (Retains feature-branch pin on `main`) |
| `LMA1980/k3-arch-phi3` | `k3-core`, `kernels-rvv` pinned to `branch = "feat/m4-epic1-llama-mistral-phi3"` | **Affected** (Retains feature-branch pin on `main`) |
| `LMA1980/k3-arch-bitnet` | `k3-core`, `kernels-rvv` pinned to `branch = "feat/m4-epic2-bitnet-tq1-tq2"` | **Affected** (Retains feature-branch pin on `main`) |
| `LMA1980/k3-arch-dory` | `k3-core`, `kernels-rvv` pinned to `branch = "main"` | Note: explicit `branch = "main"` differs from omitted `branch` in Cargo source identity |
| `LMA1980/k3-engine` | `arch-*`, `k3-core`, `k3-tokenizer`, `kernels-rvv` pinned to `branch = "feat/m4-epic1-llama-mistral-phi3"` | **Affected** (Retains feature-branch pin on `main`) |
| `LMA1980/k3-kv` | `k3-core` pinned to `branch = "feat/m4-epic1-llama-mistral-phi3"` | **Affected** (Retains feature-branch pin on `main`) |
| `LMA1980/k3-manager` | `k3-arch-mistral`, `k3-core`, `k3-tokenizer`, `kernels-rvv`, `k3-kv` pinned to `branch = "feat/m4-epic1-llama-mistral-phi3"` | **Affected** (Retains feature-branch pin on `main`) |

---

## 4. Remediation & Preventive Controls

### Immediate Remediation
1. **`LMA1980/k3-kernels-rvv` (`PR #42`)**:
   - Branched `feat/fix-main-k3-core-git-dep` from `origin/main`.
   - Removed `branch = "feat/m4-epic1-llama-mistral-phi3"` from `k3-core` in `Cargo.toml` so `k3-kernels-rvv` on `main` resolves `k3-core` via its default `main` branch (`k3-core = { version = "0.1.0", git = "https://github.com/LMA1980/k3-core.git" }`).
   - Upgraded `.actionrunner/ActionRunner.yml` to v3 with all 7 verification stages (`fmt-check`, `clippy-check`, `audit-check`, `build-release`, `test-check`, `coverage-check`, `miri-check` with `MIRIFLAGS: "-Zmiri-disable-isolation"` and `TMPDIR: "/var/lib/action-runner/.cache"`) and removed legacy `.actionrunner/on-*.yml` files.
2. **`LMA1980/k3-arch-synthetic`**:
   - Branched `feat/fix-main-ci-incident-and-v3-workflow` from `origin/main`.
   - Upgraded `.actionrunner/` from legacy `on-*.yml` to v3 `.actionrunner/ActionRunner.yml` (all 7 mandatory stages) and `.actionrunner/security.yml`.
   - Added `Symposium.toml` and `skills/k3-arch-synthetic/SKILL.md` for `cargo agents` plugin compliance.

### Preventive Controls Against Recurrence
1. **Canonical Default-Branch Git Dependency Convention on `main`**:
   - On `main`, all `Cargo.toml` git dependencies across `LMA1980/k3-*` repositories **must** omit `branch = "..."` (i.e., `git = "https://github.com/LMA1980/<repo>.git"` without `branch = "feat/..."` or `branch = "main"`), ensuring Cargo resolves a single canonical `DefaultBranch` source identity across the entire dependency graph.
2. **Pre-Merge Dependency Hygiene Gate**:
   - Before merging any `feat/**` pull request into `main` (or as part of epic merge waves in topological order: `k3-core` → `k3-tokenizer` / `k3-kernels-rvv` → `k3-arch-*` / `k3-kv` → `k3-engine` → `k3-manager`), strip temporary `branch = "feat/..."` specifiers from `Cargo.toml` and verify standalone resolution (`cargo tree --duplicates`) in an isolated worktree without parent workspace `[patch]` overrides.
3. **Mandatory 7-Stage v3 `ActionRunner.yml` on All Submodules**:
   - Enforce all 7 stages (`fmt-check`, `clippy-check`, `audit-check`, `build-release`, `test-check`, `coverage-check`, `miri-check`) on `push`, `pull_request`, and `merge` so standalone crate graph conflicts are caught prior to merge.
