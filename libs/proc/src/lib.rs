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
///    grad::when_enabled::<_>(|ctx| {
///        // original function body
///    }).expect("Gradient context required but not found.")
/// }
/// ```
/// 
/// When applied to an `impl` block, all methods within the block are wrapped similarly.
pub fn when_enabled(attr: TokenStream, item: TokenStream) -> TokenStream {
    grad::when_enabled(attr, item)
}

/// Attribute macro to only run a method when gradient tracking is enabled.
/// If gradient tracking is disabled, the method returns `None`.
/// # Usage
/// ```ignore
/// #[if_enabled]
/// fn my_function(...) -> Option<T> {
///     // ...
///     T
/// }
/// ```
/// Expands to
/// ```ignore
/// fn my_function(...) -> Option<T> {
///    grad::if_enabled::<_>(|ctx| {
///        // original function body
///    })
/// }
/// ```
#[proc_macro_attribute]
pub fn if_enabled(attr: TokenStream, item: TokenStream) -> TokenStream {
    grad::if_enabled(attr, item)
}

/// Disabled gradient tracking for the inner function.
/// This prevents routines from being tracked on the computation graph.
/// # Usage
/// ```ignore
/// #[no_grad]
/// fn my_function(...) -> ... {
///     // ...
/// }
/// ```
/// Expands to
/// ```ignore
/// fn my_function(...) -> ... {
///    grad::no_grad::<_>(|| {
///        // original function body
///    })
/// } 
/// ```
/// When applied to an `impl` block, all methods within the block are wrapped similarly
#[proc_macro_attribute]
pub fn no_grad(attr: TokenStream, item: TokenStream) -> TokenStream {
    grad::no_grad(attr, item)
}

/// Similar to `no_grad`, but provides access to the context which is being disabled.
/// # Usage
/// ```ignore
/// #[without_grad(ctx)]
/// fn my_function(...) -> Option<T> {
///     // ...
///     T
/// }
/// ```
/// Expands to
/// ```ignore
/// fn my_function(...) -> Option<T> {
///    grad::without_enabled(|ctx| {
///        // original function body
///    })
/// }
/// ```
/// Is equivelant to 
/// ```ignore
/// #[if_enabled(ctx)]
/// #[no_grad]
/// fn my_function(...) -> Option<T> {
///    // ...
///   T
/// }
/// ```
#[proc_macro_attribute]
pub fn without_grad(attr: TokenStream, item: TokenStream) -> TokenStream {
    grad::without_enabled(attr, item)
}

#[proc_macro_attribute]
/// Attribute macro to mark a function or method as incomplete in terms of gradient support.
/// When the annotated function is called while gradient tracking is enabled, a warning message
/// will be printed to standard output indicating that the function does not fully support gradients yet.
/// This allows the program to run while alerting the user to potential issues with gradient computation.
/// # Usage
/// ```ignore
/// #[incomplete]
/// fn my_function(...) -> ... {
///     // ...
/// }
/// ```
/// Expands to
/// ```ignore
/// fn my_function(...) -> ... {
///    if grad::is_enabled() {
///        eprintln!("Warning: my_function is marked as incomplete for gradient support.");
///    }
///    // original function body
/// }
/// ```
pub fn incomplete(_attr: TokenStream, item: TokenStream) -> TokenStream {
    grad::grad_incomplete(item)
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
