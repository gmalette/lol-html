use super::attribute_matcher::AttributeMatcher;
use super::program::{AddressRange, ExecutionBranch, Instruction, Program};
use super::{
    Ast, AstNode, AttributeComparisonExpr, Expr, OnAttributesExpr, OnTagNameExpr, Predicate,
    SelectorState,
};
use crate::base::{BytesCow, HasReplacementsError};
use crate::html::LocalName;
use encoding_rs::Encoding;
use selectors::attr::{AttrSelectorOperator, ParsedCaseSensitivity};
use std::iter;

type BytesOwned = Box<[u8]>;

/// An expression using only the tag name of an element.
pub type CompiledLocalNameExpr =
    Box<dyn Fn(&SelectorState<'_>, &LocalName<'_>) -> bool + Send + 'static>;
/// An expression using the attributes of an element.
pub type CompiledAttributeExpr =
    Box<dyn Fn(&SelectorState<'_>, &AttributeMatcher<'_>) -> bool + Send + 'static>;

#[derive(Default)]
struct ExprSet {
    pub local_name_exprs: Vec<CompiledLocalNameExpr>,
    pub attribute_exprs: Vec<CompiledAttributeExpr>,
}

pub(crate) struct AttrExprOperands {
    pub name: BytesOwned,
    pub value: BytesOwned,
    pub case_sensitivity: ParsedCaseSensitivity,
}

impl Expr<OnTagNameExpr> {
    #[inline]
    pub fn compile_expr<F: Fn(&SelectorState<'_>, &LocalName<'_>) -> bool + Send + 'static>(
        &self,
        f: F,
    ) -> CompiledLocalNameExpr {
        if self.negation {
            Box::new(move |s, a| !f(s, a))
        } else {
            Box::new(f)
        }
    }
}

trait Compilable {
    fn compile(
        &self,
        encoding: &'static Encoding,
        exprs: &mut ExprSet,
        enable_nth_of_type: &mut bool,
    );
}

impl Compilable for Expr<OnTagNameExpr> {
    fn compile(
        &self,
        encoding: &'static Encoding,
        exprs: &mut ExprSet,
        enable_nth_of_type: &mut bool,
    ) {
        let expr = match &self.simple_expr {
            OnTagNameExpr::ExplicitAny => self.compile_expr(|_, _| true),
            OnTagNameExpr::Unmatchable => self.compile_expr(|_, _| false),
            OnTagNameExpr::LocalName(local_name) => {
                match LocalName::from_str_without_replacements(&**local_name, encoding)
                    .map(LocalName::into_owned)
                {
                    Ok(local_name) => self.compile_expr(move |_, actual| *actual == local_name),
                    // NOTE: selector value can't be converted to the given encoding, so
                    // it won't ever match.
                    Err(_) => self.compile_expr(|_, _| false),
                }
            }
            &OnTagNameExpr::NthChild(nth) => {
                self.compile_expr(move |state, _| state.cumulative.is_nth(nth))
            }
            &OnTagNameExpr::NthOfType(nth) => {
                *enable_nth_of_type = true;
                self.compile_expr(move |state, _| {
                    state
                        .typed
                        .expect("Counter for type required at this point")
                        .is_nth(nth)
                })
            }
        };

        exprs.local_name_exprs.push(expr);
    }
}

impl Expr<OnAttributesExpr> {
    #[inline]
    pub fn compile_expr<
        F: Fn(&SelectorState<'_>, &AttributeMatcher<'_>) -> bool + Send + 'static,
    >(
        &self,
        f: F,
    ) -> CompiledAttributeExpr {
        if self.negation {
            Box::new(move |s, a| !f(s, a))
        } else {
            Box::new(f)
        }
    }
}

#[inline]
fn compile_literal(
    encoding: &'static Encoding,
    lit: &str,
) -> Result<BytesOwned, HasReplacementsError> {
    Ok(BytesCow::from_str_without_replacements(lit, encoding)?.into())
}

#[inline]
fn compile_literal_lowercase(
    encoding: &'static Encoding,
    lit: &str,
) -> Result<BytesOwned, HasReplacementsError> {
    compile_literal(encoding, &lit.to_ascii_lowercase())
}

#[inline]
fn compile_operands(
    encoding: &'static Encoding,
    name: &str,
    value: &str,
) -> Result<(BytesOwned, BytesOwned), HasReplacementsError> {
    Ok((
        compile_literal_lowercase(encoding, name)?,
        compile_literal(encoding, value)?,
    ))
}

impl Compilable for Expr<OnAttributesExpr> {
    fn compile(&self, encoding: &'static Encoding, exprs: &mut ExprSet, _: &mut bool) {
        let expr_result =
            match &self.simple_expr {
                OnAttributesExpr::Id(id) => compile_literal(encoding, id)
                    .map(|id| self.compile_expr(move |_, m| m.has_id(&id))),

                OnAttributesExpr::Class(class) => compile_literal(encoding, class)
                    .map(|class| self.compile_expr(move |_, m| m.has_class(&class))),

                OnAttributesExpr::AttributeExists(name) => compile_literal(encoding, name)
                    .map(|name| self.compile_expr(move |_, m| m.has_attribute(&name))),

                &OnAttributesExpr::AttributeComparisonExpr(AttributeComparisonExpr {
                    ref name,
                    ref value,
                    case_sensitivity,
                    operator,
                }) => compile_operands(encoding, name, value).map(move |(name, value)| {
                    let operands = AttrExprOperands {
                        name,
                        value,
                        case_sensitivity,
                    };
                    match operator {
                        AttrSelectorOperator::Equal => {
                            self.compile_expr(move |_, m| m.attr_eq(&operands))
                        }
                        AttrSelectorOperator::Includes => self
                            .compile_expr(move |_, m| m.matches_splitted_by_whitespace(&operands)),
                        AttrSelectorOperator::DashMatch => {
                            self.compile_expr(move |_, m| m.has_dash_matching_attr(&operands))
                        }
                        AttrSelectorOperator::Prefix => {
                            self.compile_expr(move |_, m| m.has_attr_with_prefix(&operands))
                        }
                        AttrSelectorOperator::Suffix => {
                            self.compile_expr(move |_, m| m.has_attr_with_suffix(&operands))
                        }
                        AttrSelectorOperator::Substring => {
                            self.compile_expr(move |_, m| m.has_attr_with_substring(&operands))
                        }
                    }
                }),
            };

        exprs
            .attribute_exprs
            .push(expr_result.unwrap_or_else(|_| self.compile_expr(|_, _| false)));
    }
}

pub(crate) struct Compiler {
    encoding: &'static Encoding,
    instructions: Box<[Option<Instruction>]>,
    free_space_start: usize,
}

impl Compiler {
    #[must_use]
    pub fn new(encoding: &'static Encoding) -> Self {
        Self {
            encoding,
            instructions: Default::default(),
            free_space_start: 0,
        }
    }

    fn compile_predicate(
        &self,
        Predicate {
            on_tag_name_exprs,
            on_attr_exprs,
        }: &Predicate,
        branch: ExecutionBranch,
        enable_nth_of_type: &mut bool,
    ) -> Instruction {
        let mut exprs = ExprSet::default();

        for c in on_tag_name_exprs {
            c.compile(self.encoding, &mut exprs, enable_nth_of_type);
        }
        for c in on_attr_exprs {
            c.compile(self.encoding, &mut exprs, enable_nth_of_type);
        }

        let ExprSet {
            local_name_exprs,
            attribute_exprs,
        } = exprs;

        debug_assert!(
            !local_name_exprs.is_empty() || !attribute_exprs.is_empty(),
            "Predicate should contain expressions"
        );

        Instruction {
            associated_branch: branch,
            local_name_exprs: local_name_exprs.into(),
            attribute_exprs: attribute_exprs.into(),
        }
    }

    /// Reserves space for a set of nodes, returning the range for the nodes to be placed
    #[inline]
    fn reserve(&mut self, nodes: &[AstNode]) -> AddressRange {
        let addr_range = self.free_space_start..self.free_space_start + nodes.len();

        self.free_space_start = addr_range.end;

        debug_assert!(self.free_space_start <= self.instructions.len());

        addr_range
    }

    #[inline]
    fn compile_descendants(
        &mut self,
        nodes: Vec<AstNode>,
        enable_nth_of_type: &mut bool,
    ) -> Option<AddressRange> {
        if nodes.is_empty() {
            None
        } else {
            Some(self.compile_nodes(nodes, enable_nth_of_type))
        }
    }

    fn compile_nodes(
        &mut self,
        nodes: Vec<AstNode>,
        enable_nth_of_type: &mut bool,
    ) -> AddressRange {
        // NOTE: we need sibling nodes to be in a contiguous region, so
        // we can reference them by range instead of vector of addresses.
        let addr_range = self.reserve(&nodes);

        for (node, position) in nodes.into_iter().zip(addr_range.clone()) {
            let branch = ExecutionBranch {
                matched_ids: node.match_ids,
                jumps: self.compile_descendants(node.children, enable_nth_of_type),
                hereditary_jumps: self.compile_descendants(node.descendants, enable_nth_of_type),
            };

            self.instructions[position] =
                Some(self.compile_predicate(&node.predicate, branch, enable_nth_of_type));
        }

        addr_range
    }

    // generic methods tend to be inlined, but this one is called from a couple of places,
    // and has cheap-to-pass non-constants args, so it won't benefit from being merged into its callers.
    // It's better to outline it, and let its callers be inlined.
    #[must_use]
    #[inline(never)]
    pub fn compile(mut self, ast: Ast) -> Program {
        let mut enable_nth_of_type = false;
        self.instructions = iter::repeat_with(|| None)
            .take(ast.cumulative_node_count)
            .collect();

        let entry_points = self.compile_nodes(ast.root, &mut enable_nth_of_type);

        Program {
            instructions: self
                .instructions
                .into_vec()
                .into_iter()
                .map(|o| o.unwrap())
                .collect(),
            entry_points,
            enable_nth_of_type,
        }
    }
}
