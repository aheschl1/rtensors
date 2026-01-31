use proc_macro::TokenStream;

mod grad;
mod rpc;

#[proc_macro_attribute]
/// Attribute macro to mark a routine as requiring an active gradient context.
/// If no gradient context is active at runtime, an error will be raised.
/// 
/// # Usage
/// 
/// ```ignore
/// #[requires_grad(ctx)]
/// fn my_function(...) -> ... {
///     // ...
/// }
/// ```
/// 
/// Expands to
/// ```ignore
/// fn my_function(...) -> ... {
///    grad::when_enabled::<T, B, _>(|ctx| {
///        // original function body
///    }).expect("Gradient context required but not found.")
/// }
/// ```
/// 
/// When applied to an `impl` block, all methods within the block are wrapped similarly.
pub fn when_enabled(attr: TokenStream, item: TokenStream) -> TokenStream {
    grad::when_enabled(attr, item)
}

#[proc_macro_attribute]
pub fn if_enabled(attr: TokenStream, item: TokenStream) -> TokenStream {
    grad::if_enabled(attr, item)
}

#[proc_macro_attribute]
pub fn no_grad(attr: TokenStream, item: TokenStream) -> TokenStream {
    grad::no_grad(attr, item)
}

// #[proc_macro_attribute]
/// Attribute macro to mark the main entry point of a program that uses gradients.
/// This macro initializes a gradient context and ensures it is available during the execution of the annotated function.
/// 
/// # Usage
/// ```ignore
/// #[grad::main(f32, MyBackend)]
/// fn main() {
///     // Your code here
/// }
/// ```
// pub fn main(attr: TokenStream, item: TokenStream) -> TokenStream {
//     grad::main(attr, item)
// }

#[proc_macro_attribute]
/// Attribute macro to generate RPC client routines for each method in an impl block.
///
/// # Usage
///
/// ```ignore
/// #[routines(MyRpcEnum)]
/// impl MyClient {
///     // ...
/// }
/// ```
///
/// Optionally, methods can be annotated with `#[rpc(skip)]` to skip codegen, or `#[rpc(extra(...))]` to add extra arguments.
/// By default, the variant for the method is derived from the method name in CamelCase.
/// To override the variant name, use `#[rpc(variant(VariantName))]`.
pub fn routines(attr: TokenStream, item: TokenStream) -> TokenStream {
    rpc::routines(attr, item)
}
