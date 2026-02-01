use proc_macro::{TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Block, Ident, Item, Lit, Meta, MetaNameValue, Result, Signature, Token};

// replaces body with grad::no_grad(|| { original body })
pub fn no_grad(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item = parse_macro_input!(item as Item);

    match item {
        Item::Fn(mut func_block) => {
            let original_block = &func_block.block;
            *func_block.block = syn::parse_quote! {
                {
                    grad::no_grad(|| {
                        #original_block
                    })
                }
            };
            quote!(#func_block).into()
        },
        Item::Impl(mut impl_block) => {
            for item in impl_block.items.iter_mut() {
                if let syn::ImplItem::Fn(method) = item {
                    let original_block = &method.block;
                    method.block = syn::parse_quote! {
                        {
                            grad::no_grad(|| {
                                #original_block
                            })
                        }
                    };
                }
            }
            quote!(#impl_block).into()
        },
        _ => {
            syn::Error::new_spanned(
                item,
                "#[no_grad] can only be applied to functions or impl blocks.",
            ).to_compile_error().into()
        }
    }
}

/// allow syntax like #[requires_grad(message = "custom error message")]
/// this is the inner function of the proc macro
pub fn when_enabled(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as GradArgs);
    let item = parse_macro_input!(item as Item);

    match item {
        Item::Fn(func_block) => {
            requires_grad_func(&args, func_block, true)
        },
        Item::Impl(impl_block) => {
            requires_grad_impl(&args, impl_block, true)
        },
        _ => {
            syn::Error::new_spanned(
                item,
                "#[requires_grad] can only be applied to functions or impl blocks.",
            ).to_compile_error().into()
        }
    }
}

/// allow syntax like #[if_enabled(ctx)]
/// this is the inner function of the proc macro
pub fn if_enabled(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as GradArgs);
    let item = parse_macro_input!(item as Item);

    if args.message.is_some() {
        return syn::Error::new_spanned(
            item,
            "#[if_enabled] does not support a custom message argument.",
        ).to_compile_error().into();
    }

    match item {
        Item::Fn(func_block) => {
            requires_grad_func(&args, func_block, false)
        },
        Item::Impl(impl_block) => {
            requires_grad_impl(&args, impl_block, false)
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
    expect: bool 
) -> syn::Result<Block> {

    let ctx_ident = &args.ctx;
    let default_failure = format!(
        "Gradient context required in method {} but not found.",
        sig.ident
    );
    let failure_msg = args.message.as_deref().unwrap_or(&default_failure);

    let expectation = if expect {
        quote! {.expect(#failure_msg) }
    } else {
        quote! {}
    };

    Ok(syn::parse_quote! {
        {
            grad::when_enabled::<_>(|#ctx_ident| {
                #block
            })
            #expectation
        }
    })
}


fn requires_grad_func(
    args: &GradArgs,
    mut func: syn::ItemFn,
    expect: bool 
) -> TokenStream {
    match make_wrapped_block(&func.sig, &func.block, args, expect) {
        Ok(new_block) => {
            *func.block = new_block;
            quote!(#func).into()
        }
        Err(e) => e.to_compile_error().into(),
    }
}


fn requires_grad_impl(
    args: &GradArgs,
    mut impl_block: syn::ItemImpl,
    expect: bool 
) -> TokenStream {
    
    for item in impl_block.items.iter_mut() {
        if let syn::ImplItem::Fn(method) = item {
            match make_wrapped_block(
                &method.sig,
                &method.block,
                args,
                expect,
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