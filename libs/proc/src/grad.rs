use proc_macro::{TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Block, Ident, Item, Lit, Meta, MetaNameValue, Result, Signature, Token};


/// allow syntax like #[requires_grad(message = "custom error message")]
/// this is the inner function of the proc macro
pub fn when_enabled(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as GradArgs);
    let item = parse_macro_input!(item as Item);

    match item {
        Item::Fn(func_block) => {
            requires_grad_func(&args, func_block)
        },
        Item::Impl(impl_block) => {
            requires_grad_impl(&args, impl_block)
        },
        _ => {
            syn::Error::new_spanned(
                item,
                "#[requires_grad] can only be applied to functions or impl blocks.",
            ).to_compile_error().into()
        }
    }
}

fn make_wrapped_block(
    sig: &Signature,
    block: &Block,
    args: &GradArgs,
    // has_t: bool,
    // has_b: bool,
) -> syn::Result<Block> {
    // if !has_t || !has_b {
    //     return Err(syn::Error::new_spanned(
    //         &sig.generics,
    //         "#[requires_grad] functions must have generic parameters T and B.",
    //     ));
    // }

    let ctx_ident = &args.ctx;
    let default_failure = format!(
        "Gradient context required in method {} but not found.",
        sig.ident
    );
    let failure_msg = args.message.as_deref().unwrap_or(&default_failure);

    Ok(syn::parse_quote! {
        {
            grad::when_enabled::<_>(|#ctx_ident| {
                #block
            })
            .expect(#failure_msg)
        }
    })
}


fn requires_grad_func(
    args: &GradArgs,
    mut func: syn::ItemFn,
    // inherited_t: bool,
    // inherited_b: bool,
) -> TokenStream {
    let generics = &func.sig.generics;

    // let has_t = inherited_t || generics.type_params().any(|tp| tp.ident == "T");
    // let has_b = inherited_b || generics.type_params().any(|tp| tp.ident == "B");

    match make_wrapped_block(&func.sig, &func.block, args) {
        Ok(new_block) => {
            func.block = Box::new(new_block);
            quote!(#func).into()
        }
        Err(e) => e.to_compile_error().into(),
    }
}


fn requires_grad_impl(
    args: &GradArgs,
    mut impl_block: syn::ItemImpl,
) -> TokenStream {
    let generics = &impl_block.generics;

    // let has_t = generics.type_params().any(|tp| tp.ident == "T");
    // let has_b = generics.type_params().any(|tp| tp.ident == "B");

    for item in impl_block.items.iter_mut() {
        if let syn::ImplItem::Fn(method) = item {
            match make_wrapped_block(
                &method.sig,
                &method.block,
                args,
                // has_t,
                // has_b,
            ) {
                Ok(new_block) => {
                    method.block = new_block;
                }
                Err(e) => {
                    return e.to_compile_error().into();
                }
            }
        }
    }

    quote!(#impl_block).into()
}



/// Arguments for the `#[requires_grad(...)]` attribute macro.
struct GradArgs {
    pub ctx: Ident,
    pub message: Option<String>
}

impl Parse for GradArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let ctx: Ident = input.parse()?;
        let mut message = None;

        if !input.is_empty() {
            input.parse::<Token![,]>()?;
        }
        
        while !input.is_empty() {
            let meta: Meta = input.parse()?;

            match meta {
                Meta::NameValue(MetaNameValue { path, value, .. }) => {
                    if path.is_ident("message") {
                        match value {
                            syn::Expr::Lit(expr) => {
                                if let Lit::Str(lit) = expr.lit {
                                    message = Some(lit.value());
                                } else {
                                    return Err(syn::Error::new_spanned(
                                        expr,
                                        "message must be a string literal",
                                    ));
                                }
                            }
                            _ => {
                                return Err(syn::Error::new_spanned(
                                    value,
                                    "message must be a string literal",
                                ));
                            }
                        }
                    } else {
                        return Err(syn::Error::new_spanned(
                            path,
                            "unknown argument",
                        ));
                    }
                }
                _ => {
                    return Err(syn::Error::new_spanned(
                        meta,
                        "expected name = \"value\"",
                    ));
                }
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(GradArgs { ctx, message })
    }
}