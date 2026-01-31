fn main() {
    // Force Cargo to recompile when CUDA kernel files change
    // Without this, include_str!() embeds file content at compile time
    // but Cargo doesn't track external file changes
    println!("cargo:rerun-if-changed=../core/gpu_backend/batch_signal_kernels.cu");
    println!("cargo:rerun-if-changed=../core/gpu_backend/batch_ma_kernels.cu");
    println!("cargo:rerun-if-changed=../core/gpu_backend/gpu_common.h");
}
