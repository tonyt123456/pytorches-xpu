//! Locating the plugin library for the integration tests.

use std::path::PathBuf;

/// The plugin cdylib. `cargo test` does not build a package's cdylib, so build it on first use into
/// the same target directory (`target/<profile>/`, one level above the test binary's `deps/`).
pub fn xpu_dll() -> PathBuf {
    let name = format!(
        "{}pytorches_plugin_xpu.{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_EXTENSION
    );
    let profile_dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let dll = profile_dir.join(&name);
    if !dll.exists() {
        let mut build =
            std::process::Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
        build
            .args(["build", "--lib", "--manifest-path"])
            .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"));
        if profile_dir.file_name().is_some_and(|n| n == "release") {
            build.arg("--release");
        }
        assert!(
            build.status().expect("run cargo").success(),
            "building the plugin failed"
        );
    }
    dll
}
