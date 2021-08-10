/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! A JEXL evaluator written in Rust
//! This crate depends on a JEXL parser crate that handles all the parsing
//! and is a part of the same workspace.
//! JEXL is an expression language used by Mozilla, you can find more information here: https://github.com/mozilla/mozjexl
//!
//! # How to use
//! The access point for this crate is the `eval` functions of the Evaluator Struct
//! You can use the `eval` function directly to evaluate standalone statements
//!
//! For example:
//! ```rust
//! use jexl_eval::Evaluator;
//! use serde_json::json as value;
//! let evaluator = Evaluator::new();
//! assert_eq!(evaluator.eval("'Hello ' + 'World'").unwrap(), value!("Hello World"));
//! ```
//!
//! You can also run the statements against a context using the `eval_in_context` function
//! The context can be any type that implements the `serde::Serializable` trait
//! and the function will return errors if the statement doesn't match the context
//!
//! For example:
//! ```rust
//! use jexl_eval::Evaluator;
//! use serde_json::json as value;
//! let context = value!({"a": {"b": 2.0}});
//! let evaluator = Evaluator::new();
//! assert_eq!(evaluator.eval_in_context("a.b", context).unwrap(), value!(2.0));
//! ```
//!

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use chrono::{LocalResult, TimeZone};

use error::*;
use jexl_parser::{
    ast::{Expression, OpCode},
    Parser,
};
use jexl_parser::ast::{ArrayValue, DateTimeValue, DateLikeValue, DurationValue, NumericValue, StdFunction, StringValue, TimeLikeValue};
pub use value::{Number, Value, DateTime, to_value};
use semver::Version;

pub mod error;
mod value;

const EPSILON: f64 = 0.000001f64;

trait Truthy {
    fn is_truthy(&self) -> bool;

    fn is_falsey(&self) -> bool {
        !self.is_truthy()
    }
}

impl Truthy for Value {
    fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Null => false,
            Value::Number(f) => f.as_f64().unwrap() != 0.0,
            Value::String(s) => !s.is_empty(),
            // It would be better if these depended on the contents of the
            // object (empty array/object is falsey, non-empty is truthy, like
            // in Python) but this matches JS semantics. Is it worth changing?
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
            Value::DateTime(_v) => true,
            Value::Date(_v) => true,
            Value::Time(_v) => true,
            Value::Duration(_v) => true,
            Value::SemVer(version) => {
                !(version.major == 0 && version.minor == 0 && version.patch == 0)
            }
        }
    }
}

type EvaluationContext = Value;

/// TransformFn represents an arbitrary transform function
/// Transform functions take an arbitrary number of `serde_json::Value`to represent their arguments
/// and return a `serde_json::Value`.
/// the transform function itself is responsible for checking if the format and number of
/// the arguments is correct
///
/// Returns a Result with an `anyhow::Error`. This allows consumers to return their own custom errors
/// in the closure, and use `.into` to convert it into an `anyhow::Error`. The error message will be perserved
pub type TransformFn = Box<dyn Fn(Option<&[Value]>) -> Result<Value, anyhow::Error> + Send + Sync>;

#[derive(Default)]
pub struct Evaluator {
    transforms: HashMap<String, TransformFn>,
}

impl Evaluator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a custom transform function
    /// This is meant as a way to allow consumers to add their own custom functionality
    /// to the expression language.
    /// Note that the name added here has to match with
    /// the name that the transform will have when it's a part of the expression statement
    ///
    /// # Arguments:
    /// - `name`: The name of the transfrom
    /// - `transform`: The actual function. A closure the implements Fn(&[serde_json::Value]) -> Result<Value, anyhow::Error>
    ///
    /// # Example:
    ///
    /// ```rust
    /// use jexl_eval::Evaluator;
    /// use serde_json::{json as value, Value};
    ///
    /// let mut evaluator = Evaluator::new().with_transform("lower", |v: &[Value]| {
    ///    let s = v
    ///            .first()
    ///            .expect("Should have 1 argument!")
    ///            .as_str()
    ///            .expect("Should be a string!");
    ///       Ok(value!(s.to_lowercase()))
    ///  });
    ///
    /// assert_eq!(evaluator.eval("'JOHN DOe'|lower").unwrap(), value!("john doe"))
    /// ```
    pub fn with_transform<F>(mut self, name: &str, transform: F) -> Self
        where
            F: Fn(Option<&[Value]>) -> Result<Value, anyhow::Error> + 'static + Send + Sync,
    {
        self.transforms
            .insert(name.to_string(), Box::new(transform));
        self
    }

    pub fn eval(&self, input: &str) -> Result<Value> {
        let context = Value::from(BTreeMap::new());
        self.eval_in_context(input, &context)
    }

    pub fn eval_in_context<T: serde::Serialize>(
        &self,
        input: &str,
        context: T,
    ) -> Result<Value> {
        let tree = Parser::parse(input)?;
        let context = value::to_value(context)?;
        if !context.is_object() {
            return Err(EvaluationError::InvalidContext);
        }
        self.eval_ast(tree, &context)
    }

    pub fn eval_ast(&self, ast: Expression, context: &EvaluationContext) -> Result<Value> {
        match ast {
            Expression::Number(n) => Ok((n).into()),
            Expression::Boolean(b) => Ok(Value::Bool(b)),
            Expression::String(s) => Ok(Value::String(s)),
            Expression::Array(xs) => xs.into_iter().map(|x| self.eval_ast(*x, context)).collect(),

            Expression::Object(items) => {
                let mut map = BTreeMap::new(/*items.len()*/);
                for (key, expr) in items.into_iter() {
                    if map.contains_key(&key) {
                        return Err(EvaluationError::DuplicateObjectKey(key));
                    }
                    let value = self.eval_ast(*expr, context)?;
                    map.insert(key, value);
                }
                Ok(Value::Object(map))
            }

            Expression::Identifier(inner) => {
                // TODO: Make this error out if the identifier does not exist in the
                // context
                Ok(context.get(&inner).unwrap_or(&Value::Null).clone())
            }

            Expression::DotOperation { subject, ident, default_value } => {
                let subject = self.eval_ast(*subject, context)?;
                let value = subject.get(&ident);
                match value {
                    None => {
                        match default_value {
                            None => {
                                Ok(Value::Null)
                            }
                            Some(default_value_expr) => {
                                self.eval_ast(*default_value_expr, context)
                            }
                        }
                    }
                    Some(value) => { Ok(value.clone()) }
                }
            }

            Expression::IndexOperation { subject, index } => {
                let subject = self.eval_ast(*subject, context)?;
                if let Expression::Filter { ident, op, right } = *index {
                    let subject_arr = subject.as_array().ok_or(EvaluationError::InvalidFilter)?;
                    let right = self.eval_ast(*right, context)?;
                    let filtered = subject_arr
                        .iter()
                        .filter(|e| {
                            let left = e.get(&ident).unwrap_or(&Value::Null);
                            // returns false if any members fail the op, could happen if array members are missing the identifier
                            Self::apply_op(op, left.clone(), right.clone())
                                .unwrap_or(Value::Bool(false))
                                .is_truthy()
                        })
                        .map(|i| i.clone())
                        .collect::<Vec<_>>();
                    return Ok(filtered.into());
                }

                let index = self.eval_ast(*index, context)?;
                match index {
                    Value::String(inner) => {
                        Ok(subject.get(&inner).unwrap_or(&Value::Null).clone())
                    }
                    Value::Number(inner) => Ok(subject
                        .get(inner.as_f64().unwrap().floor() as usize)
                        .unwrap_or(&Value::Null)
                        .clone()),
                    _ => Err(EvaluationError::InvalidIndexType),
                }
            }

            Expression::NegationOperation { expr } => {
                let value = self.eval_ast(*expr, context)?;
                Ok(Value::Bool(!value.is_truthy()))
            }

            Expression::BinaryOperation {
                left,
                right,
                operation,
            } => {
                let left = self.eval_ast(*left, context)?;
                let right = self.eval_ast(*right, context)?;
                Self::apply_op(operation, left, right)
            }
            Expression::CustomTransform {
                name,
                subject,
                args,
            } => {
                let subject = self.eval_ast(*subject, context)?;
                let mut args_arr = Vec::new();
                args_arr.push(subject);
                if let Some(args) = args {
                    for arg in args {
                        args_arr.push(self.eval_ast(*arg, context)?);
                    }
                }
                let f = self
                    .transforms
                    .get(&name)
                    .ok_or(EvaluationError::UnknownTransform(name))?;
                f(Some(&args_arr)).map_err(|e| e.into())
            }

            Expression::Conditional {
                left,
                truthy,
                falsy,
            } => {
                if self.eval_ast(*left, context)?.is_truthy() {
                    self.eval_ast(*truthy, context)
                } else {
                    self.eval_ast(*falsy, context)
                }
            }

            Expression::Filter {
                ident: _,
                op: _,
                right: _,
            } => {
                // Filters shouldn't be evaluated individually
                // instead, they are evaluated as a part of an IndexOperation
                return Err(EvaluationError::InvalidFilter);
            }
            Expression::StdFunction(std_func) => self.eval_std_func(std_func, context),
            Expression::CustomFunction { name, args } => {
                let f = self
                    .transforms
                    .get(&name)
                    .ok_or(EvaluationError::UnknownTransform(name))?;

                if let Some(args) = args {
                    let mut args_arr = Vec::new();
                    for arg in args {
                        args_arr.push(self.eval_ast(*arg, context)?);
                    }

                    f(Some(&args_arr)).map_err(|e| e.into())
                } else {
                    f(None).map_err(|e| e.into())
                }
            }
        }
    }

    fn eval_std_func(&self, std_func: StdFunction, context: &EvaluationContext) -> Result<Value> {
        use voca_rs::Voca;

        match std_func {
            StdFunction::FuncAny(subject) => {
                let array = self.eval_array_value(subject, context)?;
                if array.is_empty() {
                    return Ok(Value::Bool(false));
                }

                for element in array {
                    if element.is_truthy() {
                        return Ok(Value::Bool(true));
                    }
                }

                return Ok(Value::Bool(false));
            }
            StdFunction::FuncAll(subject) => {
                let array = self.eval_array_value(subject, context)?;
                if array.is_empty() {
                    return Ok(Value::Bool(false));
                }

                for element in array {
                    if !element.is_truthy() {
                        return Ok(Value::Bool(false));
                    }
                }

                return Ok(Value::Bool(true));
            }
            StdFunction::FuncMax(subject) => {
                let array = self.eval_array_value(subject, context)?;
                if array.is_empty() {
                    return Ok(Value::Null);
                }

                let mut max = f64::MIN;
                for element in array {
                    if let Value::Number(number) = element {
                        if let Some(number) = number.as_f64() {
                            if max < number {
                                max = number;
                            }
                        } else {
                            return Err(EvaluationError::InvalidValueType { expected: "Number".to_string(), got: format!("{}", number) });
                        }
                    } else {
                        return Err(EvaluationError::InvalidValueType { expected: "Number".to_string(), got: format!("{:?}", element) });
                    }
                }

                return Ok(max.into());
            }
            StdFunction::FuncMin(subject) => {
                let array = self.eval_array_value(subject, context)?;
                if array.is_empty() {
                    return Ok(Value::Null);
                }

                let mut min = f64::MAX;
                for element in array {
                    if let Value::Number(number) = element {
                        if let Some(number) = number.as_f64() {
                            if min > number {
                                min = number;
                            }
                        } else {
                            return Err(EvaluationError::InvalidValueType { expected: "Number".to_string(), got: format!("{}", number) });
                        }
                    } else {
                        return Err(EvaluationError::InvalidValueType { expected: "Number".to_string(), got: format!("{:?}", element) });
                    }
                }

                return Ok(min.into());
            }
            StdFunction::FuncSum(subject) => {
                let array = self.eval_array_value(subject, context)?;
                if array.is_empty() {
                    return Ok(Value::Null);
                }

                let mut sum = 0.0;
                for element in array {
                    if let Value::Number(number) = element {
                        if let Some(number) = number.as_f64() {
                            sum += number;
                        } else {
                            return Err(EvaluationError::InvalidValueType { expected: "Number".to_string(), got: format!("{}", number) });
                        }
                    } else {
                        return Err(EvaluationError::InvalidValueType { expected: "Number".to_string(), got: format!("{:?}", element) });
                    }
                }

                return Ok(sum.into());
            }
            StdFunction::FuncLen(subject) => {
                match self.eval_ast(*subject, context)? {
                    Value::Null => Ok(Value::Number(Number::from(0))),
                    Value::String(s) => Ok(Value::Number(Number::from(s.len()))),
                    Value::Bool(_) => Err(EvaluationError::InvalidValueType { expected: "String/Array/Object".to_string(), got: "Bool".to_string() }),
                    Value::Number(_) => Err(EvaluationError::InvalidValueType { expected: "String/Array/Object".to_string(), got: "Number".to_string() }),
                    Value::Array(a) => Ok(Value::Number(Number::from(a.len()))),
                    Value::Object(o) => Ok(Value::Number(Number::from(o.len()))),
                    Value::DateTime(_) => Err(EvaluationError::InvalidValueType { expected: "String/Array/Object".to_string(), got: "DateTime".to_string() }),
                    Value::Date(_) => Err(EvaluationError::InvalidValueType { expected: "String/Array/Object".to_string(), got: "Date".to_string() }),
                    Value::Time(_) => Err(EvaluationError::InvalidValueType { expected: "String/Array/Object".to_string(), got: "Time".to_string() }),
                    Value::Duration(_) => Err(EvaluationError::InvalidValueType { expected: "String/Array/Object".to_string(), got: "Duration".to_string() }),
                    Value::SemVer(_) => Err(EvaluationError::InvalidValueType { expected: "String/Array/Object".to_string(), got: "SemVer".to_string() }),
                }
            }
            StdFunction::FuncIsEmpty(subject) => {
                match self.eval_ast(*subject, context)? {
                    Value::Null => Ok(Value::Bool(true)),
                    Value::String(s) => Ok(Value::Bool(s.is_empty())),
                    Value::Bool(_) => Err(EvaluationError::InvalidValueType { expected: "String/Array/Object".to_string(), got: "Bool".to_string() }),
                    Value::Number(_) => Err(EvaluationError::InvalidValueType { expected: "String/Array/Object".to_string(), got: "Number".to_string() }),
                    Value::Array(a) => Ok(Value::Bool(a.is_empty())),
                    Value::Object(o) => Ok(Value::Bool(o.is_empty())),
                    Value::DateTime(_) => Err(EvaluationError::InvalidValueType { expected: "String/Array/Object".to_string(), got: "DateTime".to_string() }),
                    Value::Date(_) => Err(EvaluationError::InvalidValueType { expected: "String/Array/Object".to_string(), got: "Date".to_string() }),
                    Value::Time(_) => Err(EvaluationError::InvalidValueType { expected: "String/Array/Object".to_string(), got: "Time".to_string() }),
                    Value::Duration(_) => Err(EvaluationError::InvalidValueType { expected: "String/Array/Object".to_string(), got: "Duration".to_string() }),
                    Value::SemVer(_) => Err(EvaluationError::InvalidValueType { expected: "String/Array/Object".to_string(), got: "SemVer".to_string() }),
                }
            }
            StdFunction::FuncCapitalise(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::String(value._capitalize(true)))
            }
            StdFunction::FuncUpperCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::String(value._upper_case()))
            }
            StdFunction::FuncLowerCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::String(value._lower_case()))
            }
            StdFunction::FuncTitleCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::String(value._title_case()))
            }
            StdFunction::FuncKebabCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::String(value._kebab_case()))
            }
            StdFunction::FuncSnakeCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::String(value._snake_case()))
            }
            StdFunction::FuncSwapCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::String(value._swap_case()))
            }
            StdFunction::FuncTrainCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::String(value._train_case()))
            }
            StdFunction::FuncPascalCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::String(value._pascal_case()))
            }
            StdFunction::FuncShoutyKebabCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::String(value._shouty_kebab_case()))
            }
            StdFunction::FuncShoutySnakeCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::String(value._shouty_snake_case()))
            }
            StdFunction::FuncMD5(subject) => {
                let value = self.eval_string_value(subject, context)?;
                let digest = md5::compute(value.as_bytes());
                Ok(Value::String(format!("{:x}", digest)))
            }
            StdFunction::FuncSHA256(subject) => {
                let value = self.eval_string_value(subject, context)?;

                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(value.as_bytes());
                let result = hasher.finalize();
                Ok(Value::String(format!("{:x}", result)))
            }
            StdFunction::FuncStrip { subject, with } => {
                let value = self.eval_string_value(subject, context)?;

                let trimmed = match with {
                    None => {
                        value._trim("")
                    }
                    Some(with) => {
                        let with = self.eval_string_value(with, context)?;

                        value._trim(&with)
                    }
                };

                Ok(Value::String(trimmed))
            }
            StdFunction::FuncLStrip { subject, with } => {
                let value = self.eval_string_value(subject, context)?;

                let trimmed = match with {
                    None => {
                        value._trim_left("")
                    }
                    Some(with) => {
                        let with = self.eval_string_value(with, context)?;
                        value._trim_left(&with)
                    }
                };

                Ok(Value::String(trimmed))
            }
            StdFunction::FuncRStrip { subject, with } => {
                let value = self.eval_string_value(subject, context)?;

                let trimmed = match with {
                    None => {
                        value._trim_right("")
                    }
                    Some(with) => {
                        let with = self.eval_string_value(with, context)?;
                        value._trim_right(&with)
                    }
                };

                Ok(Value::String(trimmed))
            }
            StdFunction::FuncEndsWith { subject, with, start, end } => {
                let value = self.eval_string_value(subject, context)?;
                let with = self.eval_string_value(with, context)?;

                let range = self.slice_range(&value, start, end, context)?;

                Ok(Value::Bool(value[range]._ends_with(&with)))
            }
            StdFunction::FuncStartsWith { subject, with, start, end } => {
                let value = self.eval_string_value(subject, context)?;
                let with = self.eval_string_value(with, context)?;

                let range = self.slice_range(&value, start, end, context)?;

                Ok(Value::Bool(value[range]._starts_with(&with)))
            }
            StdFunction::FuncIsDecimal(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._is_numeric()))
            }
            StdFunction::FuncIsAlphaNum(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._is_alphadigit()))
            }
            StdFunction::FuncIsAlpha(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._is_alpha()))
            }
            StdFunction::FuncIsDigit(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._is_digit()))
            }
            StdFunction::FuncIsNumeric(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._is_numeric()))
            }
            StdFunction::FuncIsSpace(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._matches(r"[\s]*", 0)))
            }
            StdFunction::FuncIsIdentifier(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._matches(r"[a-zA-Z_][a-zA-Z0-9_]*", 0)))
            }
            StdFunction::FuncIsLowerCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._is_lowercase()))
            }
            StdFunction::FuncIsUpperCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._is_uppercase()))
            }
            StdFunction::FuncIsTitleCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._is_title()))
            }
            StdFunction::FuncIsCapitalise(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._is_capitalize()))
            }
            StdFunction::FuncIsKebabCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._is_kebab_case()))
            }
            StdFunction::FuncIsSnakeCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._is_snake_case()))
            }
            StdFunction::FuncIsTrainCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._is_train_case()))
            }
            StdFunction::FuncIsPascalCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._is_pascal_case()))
            }
            StdFunction::FuncIsShoutyKebabCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._is_shouty_kebab_case()))
            }
            StdFunction::FuncIsShoutySnakeCase(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(Value::Bool(value._is_shouty_snake_case()))
            }
            StdFunction::FuncSplit { subject, with, num_splits } => {
                let value = self.eval_string_value(subject, context)?;
                let with = self.eval_string_value(with, context)?;

                let num_splits = self.eval_to_usize(num_splits, context, || 0)?;
                let splits = if num_splits > 0 {
                    value.splitn(num_splits, &with).collect::<Vec<&str>>()
                } else {
                    value.split(&with).collect::<Vec<&str>>()
                };

                Ok(splits.into())
            }
            StdFunction::FuncRSplit { subject, with, num_splits } => {
                let value = self.eval_string_value(subject, context)?;
                let with = self.eval_string_value(with, context)?;

                let num_splits = self.eval_to_usize(num_splits, context, || 0)?;
                let splits = if num_splits > 0 {
                    value.rsplitn(num_splits, &with).collect::<Vec<&str>>()
                } else {
                    value.rsplit(&with).collect::<Vec<&str>>()
                };

                Ok(splits.into())
            }
            StdFunction::FuncWords(subject) => {
                let value = self.eval_string_value(subject, context)?;
                Ok(value._words().into())
            }
            StdFunction::FuncIndex { subject, search, start, end } => {
                let value = self.eval_string_value(subject, context)?;
                let search = self.eval_string_value(search, context)?;

                let range = self.slice_range(&value, start, end, context)?;

                Ok(Value::Number(Number::from(value[range]._index_of(&search, 0))))
            }
            StdFunction::FuncRIndex { subject, search, start, end } => {
                let value = self.eval_string_value(subject, context)?;
                let search = self.eval_string_value(search, context)?;

                let range = self.slice_range(&value, start, end, context)?;

                Ok(Value::Number(Number::from(value[range]._last_index_of(&search, 0))))
            }
            StdFunction::FuncFind { subject, search, start, end } => {
                let value = self.eval_string_value(subject, context)?;
                let search = self.eval_string_value(search, context)?;

                let range = self.slice_range(&value, start, end, context)?;

                Ok(Value::Number(Number::from(value[range]._index_of(&search, 0))))
            }
            StdFunction::FuncRFind { subject, search, start, end } => {
                let value = self.eval_string_value(subject, context)?;
                let search = self.eval_string_value(search, context)?;

                let range = self.slice_range(&value, start, end, context)?;

                Ok(Value::Number(Number::from(value[range]._last_index_of(&search, 0))))
            }
            StdFunction::FuncAbs(subject) => {
                let value = self.eval_numeric_value(subject, context)?;
                if value.is_f64() {
                    let value = value.as_f64().unwrap();
                    Ok(value.abs().into())
                } else if value.is_i64() {
                    let value = value.as_i64().unwrap();
                    Ok(value.abs().into())
                } else if value.is_u64() {
                    let value = value.as_u64().unwrap();
                    Ok(value.into())
                } else {
                    Ok(Value::Null)
                }
            }
            StdFunction::FuncRound(subject, num_digits) => {
                let value = self.eval_numeric_value(subject, context)?;
                if value.is_f64() {
                    let value = value.as_f64().unwrap();

                    let scale = self.eval_to_usize(num_digits, context, || 0)?;
                    let multiplier = 10i64.pow(scale as u32) as f64;
                    let rounded_value = (value * multiplier).round() / multiplier;

                    Ok(rounded_value.into())
                } else if value.is_i64() {
                    let value = value.as_i64().unwrap();
                    Ok(value.into())
                } else if value.is_u64() {
                    let value = value.as_u64().unwrap();
                    Ok(value.into())
                } else {
                    Ok(Value::Null)
                }
            }
            StdFunction::FuncHex(subject) => {
                let value = self.eval_numeric_value(subject, context)?;
                if value.is_f64() {
                    Err(EvaluationError::InvalidValueType { expected: "Integer".to_string(), got: "Float".to_string() })
                } else if value.is_i64() {
                    let value = value.as_i64().unwrap();
                    Ok(Value::String(format!("{:x}", value)))
                } else if value.is_u64() {
                    let value = value.as_u64().unwrap();
                    Ok(Value::String(format!("{:x}", value)))
                } else {
                    Ok(Value::Null)
                }
            }
            StdFunction::FuncOct(subject) => {
                let value = self.eval_numeric_value(subject, context)?;
                if value.is_f64() {
                    Err(EvaluationError::InvalidValueType { expected: "Integer".to_string(), got: "Float".to_string() })
                } else if value.is_i64() {
                    let value = value.as_i64().unwrap();
                    Ok(Value::String(format!("{:o}", value)))
                } else if value.is_u64() {
                    let value = value.as_u64().unwrap();
                    Ok(Value::String(format!("{:o}", value)))
                } else {
                    Ok(Value::Null)
                }
            }
            StdFunction::FuncParseLocalDateTime(subject) => {
                use anyhow::Context;

                let value = self.eval_string_value(subject, context)?;
                let datetime = value.parse::<chrono::DateTime<chrono::Local>>().with_context(|| format!("Error in parsing value={} to Local DateTime", value))?;
                Ok(Value::from(datetime))
            }
            StdFunction::FuncParseUtcDateTime(subject) => {
                use anyhow::Context;

                let value = self.eval_string_value(subject, context)?;
                let datetime = value.parse::<chrono::DateTime<chrono::Utc>>().with_context(|| format!("Error in parsing value={} to Utc DateTime", value))?;
                Ok(Value::from(datetime))
            }
            StdFunction::FuncLocalDateTimeFromTimestampSecs(subject) => {
                let value = self.eval_to_i64(Some(subject), context, || 0)?;
                match chrono::Local.timestamp_opt(value, 0) {
                    LocalResult::Single(datetime) => Ok(Value::from(datetime)),
                    _ => Err(EvaluationError::CustomError(anyhow::anyhow!("Error in converting timestamp value={}s to Utc DateTime", value)))
                }
            }
            StdFunction::FuncUtcDateTimeFromTimestampSecs(subject) => {
                let value = self.eval_to_i64(Some(subject), context, || 0)?;
                match chrono::Utc.timestamp_opt(value, 0) {
                    LocalResult::Single(datetime) => Ok(Value::from(datetime)),
                    _ => Err(EvaluationError::CustomError(anyhow::anyhow!("Error in converting timestamp value={}s to Local DateTime", value)))
                }
            }
            StdFunction::FuncLocalDateTimeFromTimestampMillis(subject) => {
                let value = self.eval_to_i64(Some(subject), context, || 0)?;
                match chrono::Local.timestamp_millis_opt(value) {
                    LocalResult::Single(datetime) => Ok(Value::from(datetime)),
                    _ => Err(EvaluationError::CustomError(anyhow::anyhow!("Error in converting timestamp value={}s to Utc DateTime", value)))
                }
            }
            StdFunction::FuncUtcDateTimeFromTimestampMillis(subject) => {
                let value = self.eval_to_i64(Some(subject), context, || 0)?;
                match chrono::Utc.timestamp_millis_opt(value) {
                    LocalResult::Single(datetime) => Ok(Value::from(datetime)),
                    _ => Err(EvaluationError::CustomError(anyhow::anyhow!("Error in converting timestamp value={}s to Local DateTime", value)))
                }
            }
            StdFunction::FuncDuration {
                days,
                hours,
                minutes,
                seconds,
                milliseconds,
                microseconds,
                weeks
            } => {
                let days = self.eval_to_i64(days, context, || 0)?;
                let hours = self.eval_to_i64(hours, context, || 0)?;
                let minutes = self.eval_to_i64(minutes, context, || 0)?;
                let seconds = self.eval_to_i64(seconds, context, || 0)?;
                let milliseconds = self.eval_to_i64(milliseconds, context, || 0)?;
                let microseconds = self.eval_to_i64(microseconds, context, || 0)?;
                let weeks = self.eval_to_i64(weeks, context, || 0)?;

                let total_value = (((((weeks * 7 + days) * 24 + hours) * 60 + minutes) * 60 + seconds) * 1000 + milliseconds) * 1000 + microseconds;

                Ok(Value::Duration(chrono::Duration::microseconds(total_value)))
            }
            StdFunction::FuncLocalNow() => {
                Ok(Value::from(chrono::Local::now()))
            }
            StdFunction::FuncUtcNow() => {
                Ok(Value::from(chrono::Utc::now()))
            }
            StdFunction::FuncUtcDateTime {
                y,
                m,
                d,
                h,
                mm,
                ss,
                ms,
                us
            } => {
                let y = self.eval_to_i64(y, context, || 0)? as i32;
                let m = self.eval_to_usize(m, context, || 0)? as u32;
                let d = self.eval_to_usize(d, context, || 0)? as u32;
                let h = self.eval_to_usize(h, context, || 0)? as u32;
                let mm = self.eval_to_usize(mm, context, || 0)? as u32;
                let ss = self.eval_to_usize(ss, context, || 0)? as u32;
                let ms = self.eval_to_usize(ms, context, || 0)? as u32;
                let us = self.eval_to_usize(us, context, || 0)? as u32;

                let datetime = chrono::Utc.ymd(y, m, d).and_hms_micro(h, mm, ss, ms * 1000 + us);

                Ok(Value::from(datetime))
            }
            StdFunction::FuncLocalDateTime {
                y,
                m,
                d,
                h,
                mm,
                ss,
                ms,
                us
            } => {
                let y = self.eval_to_i64(y, context, || 0)? as i32;
                let m = self.eval_to_usize(m, context, || 0)? as u32;
                let d = self.eval_to_usize(d, context, || 0)? as u32;
                let h = self.eval_to_usize(h, context, || 0)? as u32;
                let mm = self.eval_to_usize(mm, context, || 0)? as u32;
                let ss = self.eval_to_usize(ss, context, || 0)? as u32;
                let ms = self.eval_to_usize(ms, context, || 0)? as u32;
                let us = self.eval_to_usize(us, context, || 0)? as u32;

                let datetime = chrono::Local.ymd(y, m, d).and_hms_micro(h, mm, ss, ms * 1000 + us);

                Ok(Value::from(datetime))
            }
            StdFunction::FuncDate {
                y,
                m,
                d
            } => {
                let y = self.eval_to_i64(y, context, || 0)? as i32;
                let m = self.eval_to_usize(m, context, || 0)? as u32;
                let d = self.eval_to_usize(d, context, || 0)? as u32;

                Ok(Value::from(chrono::NaiveDate::from_ymd(y, m, d)))
            }
            StdFunction::FuncTime {
                h,
                mm,
                ss,
                ms,
                us
            } => {
                let h = self.eval_to_usize(h, context, || 0)? as u32;
                let mm = self.eval_to_usize(mm, context, || 0)? as u32;
                let ss = self.eval_to_usize(ss, context, || 0)? as u32;
                let ms = self.eval_to_usize(ms, context, || 0)? as u32;
                let us = self.eval_to_usize(us, context, || 0)? as u32;

                let time = chrono::NaiveTime::from_hms_micro(h, mm, ss, ms * 1000 + us);

                Ok(Value::from(time))
            }
            StdFunction::FuncGetYearFromDate(date) => {
                let date = self.eval_datelike_value(date, context)?;
                Ok(Value::from(date.year()))
            }
            StdFunction::FuncGetMonthFromDate(date) => {
                let date = self.eval_datelike_value(date, context)?;
                Ok(Value::from(date.month()))
            }
            StdFunction::FuncGetDayOfMonthFromDate(date) => {
                let date = self.eval_datelike_value(date, context)?;
                Ok(Value::from(date.day()))
            }
            StdFunction::FuncGetDayOfYearFromDate(date) => {
                let date = self.eval_datelike_value(date, context)?;
                Ok(Value::from(date.ordinal()))
            }
            StdFunction::FuncGetDayOfWeekFromDate(date) => {
                let date = self.eval_datelike_value(date, context)?;
                Ok(Value::from(date.weekday().to_string()))
            }
            StdFunction::FuncGetHourFromTime(time) => {
                let time = self.eval_timelike_value(time, context)?;
                Ok(Value::from(time.hour()))
            }
            StdFunction::FuncGetMinuteFromTime(time) => {
                let time = self.eval_timelike_value(time, context)?;
                Ok(Value::from(time.minute()))
            }
            StdFunction::FuncGetSecondFromTime(time) => {
                let time = self.eval_timelike_value(time, context)?;
                Ok(Value::from(time.second()))
            }
            StdFunction::FuncGetMillisecondFromTime(time) => {
                let time = self.eval_timelike_value(time, context)?;
                Ok(Value::from(time.millisecond()))
            }
            StdFunction::FuncGetDateFromDateTime(datetime) => {
                let datetime = self.eval_datetime_value(datetime, context)?;
                Ok(Value::from(datetime.date()))
            }
            StdFunction::FuncGetTimeFromDateTime(datetime) => {
                let datetime = self.eval_datetime_value(datetime, context)?;
                Ok(Value::from(datetime.time()))
            }
            StdFunction::FuncGetTimestampFromDateTime(datetime) => {
                let datetime = self.eval_datetime_value(datetime, context)?;
                Ok(Value::from(datetime.timestamp()))
            }
            StdFunction::FuncGetTimestampMillisFromDateTime(datetime) => {
                let datetime = self.eval_datetime_value(datetime, context)?;
                Ok(Value::from(datetime.timestamp_millis()))
            }
            StdFunction::FuncGetWeeksFromDuration(duration) => {
                let duration = self.eval_duration_value(duration, context)?;
                Ok(Value::from(duration.num_weeks()))
            }
            StdFunction::FuncGetDaysFromDuration(duration) => {
                let duration = self.eval_duration_value(duration, context)?;
                Ok(Value::from(duration.num_days()))
            }
            StdFunction::FuncGetHoursFromDuration(duration) => {
                let duration = self.eval_duration_value(duration, context)?;
                Ok(Value::from(duration.num_hours()))
            }
            StdFunction::FuncGetMinutesFromDuration(duration) => {
                let duration = self.eval_duration_value(duration, context)?;
                Ok(Value::from(duration.num_minutes()))
            }
            StdFunction::FuncGetSecondsFromDuration(duration) => {
                let duration = self.eval_duration_value(duration, context)?;
                Ok(Value::from(duration.num_seconds()))
            }
            StdFunction::FuncGetMillisecondsFromDuration(duration) => {
                let duration = self.eval_duration_value(duration, context)?;
                Ok(Value::from(duration.num_milliseconds()))
            }
            StdFunction::FuncGetMicrosecondsFromDuration(duration) => {
                let duration = self.eval_duration_value(duration, context)?;
                Ok(Value::from(duration.num_microseconds().unwrap_or_else(|| 0)))
            }
            StdFunction::FuncSemVersion(subject) => {
                let value = self.eval_string_value(subject, context)?;

                use anyhow::Context;
                let version = semver::Version::parse(&value).with_context(|| format!("Error in parsing {} to semVersion", value))?;

                Ok(Value::SemVer(version))
            }
            StdFunction::FuncParseDuration(subject) => {
                let value = self.eval_string_value(subject, context)?;

                use anyhow::Context;
                let std_duration = humantime::parse_duration(&value).with_context(|| format!("Error in parsing {} to Duration", value))?;
                let duration = chrono::Duration::from_std(std_duration).with_context(|| format!("Error in converting {} to chrono::Duration", value))?;

                Ok(Value::Duration(duration))
            }
        }
    }

    fn slice_range(&self, value: &str, start: Option<NumericValue>, end: Option<NumericValue>, context: &EvaluationContext) -> Result<Range<usize>> {
        let start_idx = self.eval_to_usize(start, context, || 0)?;
        let end_idx = self.eval_to_usize(end, context, || value.len())?;
        Ok(start_idx..end_idx)
    }

    fn eval_to_usize<F>(&self, value_expr: Option<NumericValue>, context: &EvaluationContext, default_value: F) -> anyhow::Result<usize> where F: FnOnce() -> usize {
        match value_expr {
            None => Ok(default_value()),
            Some(value_expr) => {
                let value = self.eval_numeric_value(value_expr, context)?;

                if value.is_f64() {
                    Err(anyhow::anyhow!("Expected Integer but found float"))
                } else if value.is_i64() {
                    Ok(value.as_i64().unwrap() as usize)
                } else if value.is_u64() {
                    Ok(value.as_u64().unwrap() as usize)
                } else {
                    Err(anyhow::anyhow!("Unknown number type found"))
                }
            }
        }
    }

    fn eval_to_i64<F>(&self, value_expr: Option<NumericValue>, context: &EvaluationContext, default_value: F) -> anyhow::Result<i64> where F: FnOnce() -> i64 {
        match value_expr {
            None => Ok(default_value()),
            Some(value_expr) => {
                let value = self.eval_numeric_value(value_expr, context)?;

                if value.is_f64() {
                    Err(anyhow::anyhow!("Expected Integer but found float"))
                } else if value.is_i64() {
                    Ok(value.as_i64().unwrap())
                } else if value.is_u64() {
                    Ok(value.as_u64().unwrap() as i64)
                } else {
                    Err(anyhow::anyhow!("Unknown number type found"))
                }
            }
        }
    }

    fn eval_array_value(&self, value_expr: ArrayValue, context: &EvaluationContext) -> Result<Vec<Value>> {
        match self.eval_ast(*value_expr.array, context)? {
            Value::Null => Ok(vec![]),
            Value::String(_value) => Err(EvaluationError::InvalidValueType { expected: "Array".to_string(), got: "String".to_string() }),
            Value::Bool(_) => Err(EvaluationError::InvalidValueType { expected: "Array".to_string(), got: "Bool".to_string() }),
            Value::Number(_) => Err(EvaluationError::InvalidValueType { expected: "Array".to_string(), got: "Number".to_string() }),
            Value::Array(array) => Ok(array),
            Value::Object(_) => Err(EvaluationError::InvalidValueType { expected: "Array".to_string(), got: "Object".to_string() }),
            Value::DateTime(_) => Err(EvaluationError::InvalidValueType { expected: "Array".to_string(), got: "DateTime".to_string() }),
            Value::Date(_) => Err(EvaluationError::InvalidValueType { expected: "Array".to_string(), got: "Date".to_string() }),
            Value::Time(_) => Err(EvaluationError::InvalidValueType { expected: "Array".to_string(), got: "Time".to_string() }),
            Value::Duration(_) => Err(EvaluationError::InvalidValueType { expected: "Array".to_string(), got: "Duration".to_string() }),
            Value::SemVer(_) => Err(EvaluationError::InvalidValueType { expected: "Array".to_string(), got: "SemVer".to_string() }),
        }
    }

    fn eval_string_value(&self, value_expr: StringValue, context: &EvaluationContext) -> Result<String> {
        match self.eval_ast(*value_expr.value, context)? {
            Value::Null => Ok("".to_string()),
            Value::String(value) => Ok(value),
            Value::Bool(_) => Err(EvaluationError::InvalidValueType { expected: "String".to_string(), got: "Bool".to_string() }),
            Value::Number(_) => Err(EvaluationError::InvalidValueType { expected: "String".to_string(), got: "Number".to_string() }),
            Value::Array(_) => Err(EvaluationError::InvalidValueType { expected: "String".to_string(), got: "Array".to_string() }),
            Value::Object(_) => Err(EvaluationError::InvalidValueType { expected: "String".to_string(), got: "Object".to_string() }),
            Value::DateTime(_) => Err(EvaluationError::InvalidValueType { expected: "String".to_string(), got: "DateTime".to_string() }),
            Value::Date(_) => Err(EvaluationError::InvalidValueType { expected: "String".to_string(), got: "Date".to_string() }),
            Value::Time(_) => Err(EvaluationError::InvalidValueType { expected: "String".to_string(), got: "Time".to_string() }),
            Value::Duration(_) => Err(EvaluationError::InvalidValueType { expected: "String".to_string(), got: "Duration".to_string() }),
            Value::SemVer(_) => Err(EvaluationError::InvalidValueType { expected: "String".to_string(), got: "SemVer".to_string() }),
        }
    }

    fn eval_numeric_value(&self, value_expr: NumericValue, context: &EvaluationContext) -> Result<Number> {
        match self.eval_ast(*value_expr.value, context)? {
            Value::Null => Ok(Number::from(0)),
            Value::Number(value) => Ok(value),
            Value::String(_) => Err(EvaluationError::InvalidValueType { expected: "Number".to_string(), got: "Bool".to_string() }),
            Value::Bool(_) => Err(EvaluationError::InvalidValueType { expected: "Number".to_string(), got: "Bool".to_string() }),
            Value::Array(_) => Err(EvaluationError::InvalidValueType { expected: "Number".to_string(), got: "Array".to_string() }),
            Value::Object(_) => Err(EvaluationError::InvalidValueType { expected: "Number".to_string(), got: "Object".to_string() }),
            Value::DateTime(_) => Err(EvaluationError::InvalidValueType { expected: "Number".to_string(), got: "DateTime".to_string() }),
            Value::Date(_) => Err(EvaluationError::InvalidValueType { expected: "Number".to_string(), got: "Date".to_string() }),
            Value::Time(_) => Err(EvaluationError::InvalidValueType { expected: "Number".to_string(), got: "Time".to_string() }),
            Value::Duration(_) => Err(EvaluationError::InvalidValueType { expected: "Number".to_string(), got: "Duration".to_string() }),
            Value::SemVer(_) => Err(EvaluationError::InvalidValueType { expected: "Number".to_string(), got: "SemVer".to_string() }),
        }
    }

    fn eval_datetime_value(&self, value_expr: DateTimeValue, context: &EvaluationContext) -> Result<value::DateTime> {
        match self.eval_ast(*value_expr.value, context)? {
            Value::Null => Err(EvaluationError::InvalidValueType { expected: "DateTime".to_string(), got: "Null".to_string() }),
            Value::Number(_) => Err(EvaluationError::InvalidValueType { expected: "DateTime".to_string(), got: "Number".to_string() }),
            Value::String(_) => Err(EvaluationError::InvalidValueType { expected: "DateTime".to_string(), got: "Bool".to_string() }),
            Value::Bool(_) => Err(EvaluationError::InvalidValueType { expected: "DateTime".to_string(), got: "Bool".to_string() }),
            Value::Array(_) => Err(EvaluationError::InvalidValueType { expected: "DateTime".to_string(), got: "Array".to_string() }),
            Value::Object(_) => Err(EvaluationError::InvalidValueType { expected: "DateTime".to_string(), got: "Object".to_string() }),
            Value::DateTime(datetime) => Ok(datetime),
            Value::Date(_) => Err(EvaluationError::InvalidValueType { expected: "DateTime".to_string(), got: "Date".to_string() }),
            Value::Time(_) => Err(EvaluationError::InvalidValueType { expected: "DateTime".to_string(), got: "Time".to_string() }),
            Value::Duration(_) => Err(EvaluationError::InvalidValueType { expected: "DateTime".to_string(), got: "Duration".to_string() }),
            Value::SemVer(_) => Err(EvaluationError::InvalidValueType { expected: "DateTime".to_string(), got: "SemVer".to_string() }),
        }
    }

    fn eval_datelike_value(&self, value_expr: DateLikeValue, context: &EvaluationContext) -> Result<value::DateLike> {
        match self.eval_ast(*value_expr.value, context)? {
            Value::Null => Err(EvaluationError::InvalidValueType { expected: "Date".to_string(), got: "Null".to_string() }),
            Value::Number(_) => Err(EvaluationError::InvalidValueType { expected: "Date".to_string(), got: "Number".to_string() }),
            Value::String(_) => Err(EvaluationError::InvalidValueType { expected: "Date".to_string(), got: "Bool".to_string() }),
            Value::Bool(_) => Err(EvaluationError::InvalidValueType { expected: "Date".to_string(), got: "Bool".to_string() }),
            Value::Array(_) => Err(EvaluationError::InvalidValueType { expected: "Date".to_string(), got: "Array".to_string() }),
            Value::Object(_) => Err(EvaluationError::InvalidValueType { expected: "Date".to_string(), got: "Object".to_string() }),
            Value::DateTime(datetime) => match datetime {
                DateTime::Local(dt) => Ok(value::DateLike::LocalDateTime(dt)),
                DateTime::Utc(dt) => Ok(value::DateLike::UtcDateTime(dt)),
            }
            Value::Date(dt) => Ok(value::DateLike::Date(dt)),
            Value::Time(_) => Err(EvaluationError::InvalidValueType { expected: "Date".to_string(), got: "Time".to_string() }),
            Value::Duration(_) => Err(EvaluationError::InvalidValueType { expected: "Date".to_string(), got: "Duration".to_string() }),
            Value::SemVer(_) => Err(EvaluationError::InvalidValueType { expected: "Date".to_string(), got: "SemVer".to_string() }),
        }
    }

    fn eval_timelike_value(&self, value_expr: TimeLikeValue, context: &EvaluationContext) -> Result<value::TimeLike>  {
        match self.eval_ast(*value_expr.value, context)? {
            Value::Null => Err(EvaluationError::InvalidValueType { expected: "Time".to_string(), got: "Null".to_string() }),
            Value::Number(_) => Err(EvaluationError::InvalidValueType { expected: "Time".to_string(), got: "Number".to_string() }),
            Value::String(_) => Err(EvaluationError::InvalidValueType { expected: "Time".to_string(), got: "Bool".to_string() }),
            Value::Bool(_) => Err(EvaluationError::InvalidValueType { expected: "Time".to_string(), got: "Bool".to_string() }),
            Value::Array(_) => Err(EvaluationError::InvalidValueType { expected: "Time".to_string(), got: "Array".to_string() }),
            Value::Object(_) => Err(EvaluationError::InvalidValueType { expected: "Time".to_string(), got: "Object".to_string() }),
            Value::DateTime(datetime) => match datetime {
                DateTime::Local(dt) => Ok(value::TimeLike::LocalDateTime(dt)),
                DateTime::Utc(dt) => Ok(value::TimeLike::UtcDateTime(dt)),
            }
            Value::Date(_) => Err(EvaluationError::InvalidValueType { expected: "Time".to_string(), got: "Date".to_string() }),
            Value::Time(time) => Ok(value::TimeLike::Time(time)),
            Value::Duration(_) => Err(EvaluationError::InvalidValueType { expected: "Time".to_string(), got: "Duration".to_string() }),
            Value::SemVer(_) => Err(EvaluationError::InvalidValueType { expected: "Time".to_string(), got: "SemVer".to_string() }),
        }
    }

    fn eval_duration_value(&self, value_expr: DurationValue, context: &EvaluationContext) -> Result<chrono::Duration> {
        match self.eval_ast(*value_expr.value, context)? {
            Value::Null => Err(EvaluationError::InvalidValueType { expected: "Duration".to_string(), got: "Null".to_string() }),
            Value::Number(_) => Err(EvaluationError::InvalidValueType { expected: "Duration".to_string(), got: "Number".to_string() }),
            Value::String(_) => Err(EvaluationError::InvalidValueType { expected: "Duration".to_string(), got: "Bool".to_string() }),
            Value::Bool(_) => Err(EvaluationError::InvalidValueType { expected: "Duration".to_string(), got: "Bool".to_string() }),
            Value::Array(_) => Err(EvaluationError::InvalidValueType { expected: "Duration".to_string(), got: "Array".to_string() }),
            Value::Object(_) => Err(EvaluationError::InvalidValueType { expected: "Duration".to_string(), got: "Object".to_string() }),
            Value::DateTime(_) => Err(EvaluationError::InvalidValueType { expected: "Duration".to_string(), got: "DateTime".to_string() }),
            Value::Date(_) => Err(EvaluationError::InvalidValueType { expected: "Duration".to_string(), got: "Date".to_string() }),
            Value::Time(_) => Err(EvaluationError::InvalidValueType { expected: "Duration".to_string(), got: "Time".to_string() }),
            Value::Duration(duration) => Ok(duration),
            Value::SemVer(_) => Err(EvaluationError::InvalidValueType { expected: "Duration".to_string(), got: "SemVer".to_string() }),
        }
    }

    fn apply_op(operation: OpCode, left: Value, right: Value) -> Result<Value> {
        match (operation, left, right) {
            (OpCode::And, a, b) => Ok(if a.is_truthy() { b } else { a }),
            (OpCode::Or, a, b) => Ok(if a.is_truthy() { a } else { b }),

            (op, Value::Number(a), Value::Number(b)) => {
                let left = a.as_f64().unwrap();
                let right = b.as_f64().unwrap();

                // TODO: handle divide by zero better
                Ok(match op {
                    OpCode::Add => (left + right).into(),
                    OpCode::Subtract => (left - right).into(),
                    OpCode::Multiply => (left * right).into(),
                    OpCode::Divide => (left / right).into(),
                    OpCode::FloorDivide => ((left / right).floor()).into(),
                    OpCode::Modulus => (left % right).into(),
                    OpCode::Exponent => (left.powf(right)).into(),
                    OpCode::Less => Value::Bool(left < right),
                    OpCode::Greater => Value::Bool(left > right),
                    OpCode::LessEqual => Value::Bool(left <= right),
                    OpCode::GreaterEqual => Value::Bool(left >= right),
                    OpCode::Equal => Value::Bool((left - right).abs() < EPSILON),
                    OpCode::NotEqual => Value::Bool((left - right).abs() > EPSILON),
                    OpCode::In => Value::Bool(false),
                    OpCode::And | OpCode::Or => {
                        unreachable!("Covered by previous case in parent match")
                    }
                })
            }

            (op, Value::DateTime(a), Value::DateTime(b)) => {
                match (a, b) {
                    (value::DateTime::Local(a), value::DateTime::Local(b)) => {
                        match op {
                            OpCode::Subtract => Ok((a - b).into()),
                            OpCode::Less => Ok(Value::Bool(a < b)),
                            OpCode::Greater => Ok(Value::Bool(a > b)),
                            OpCode::LessEqual => Ok(Value::Bool(a <= b)),
                            OpCode::GreaterEqual => Ok(Value::Bool(a >= b)),
                            OpCode::Equal => Ok(Value::Bool(a == b)),
                            OpCode::NotEqual => Ok(Value::Bool(a != b)),
                            _ => {
                                Err(EvaluationError::InvalidBinaryOp {
                                    operation: op,
                                    left: Value::from(a),
                                    right: Value::from(b),
                                })
                            }
                        }
                    }
                    (value::DateTime::Utc(a), value::DateTime::Utc(b)) => {
                        match op {
                            OpCode::Subtract => Ok((a - b).into()),
                            OpCode::Less => Ok(Value::Bool(a < b)),
                            OpCode::Greater => Ok(Value::Bool(a > b)),
                            OpCode::LessEqual => Ok(Value::Bool(a <= b)),
                            OpCode::GreaterEqual => Ok(Value::Bool(a >= b)),
                            OpCode::Equal => Ok(Value::Bool(a == b)),
                            OpCode::NotEqual => Ok(Value::Bool(a != b)),
                            _ => {
                                Err(EvaluationError::InvalidBinaryOp {
                                    operation: op,
                                    left: Value::from(a),
                                    right: Value::from(b),
                                })
                            }
                        }
                    }
                    (value::DateTime::Local(a), value::DateTime::Utc(b)) => {
                        Err(EvaluationError::InvalidBinaryOp {
                            operation,
                            left: Value::from(a),
                            right: Value::from(b),
                        })
                    }
                    (value::DateTime::Utc(a), value::DateTime::Local(b)) => {
                        Err(EvaluationError::InvalidBinaryOp {
                            operation,
                            left: Value::from(a),
                            right: Value::from(b),
                        })
                    }
                }
            }

            (op, Value::Duration(a), Value::Duration(b)) => {
                match op {
                    OpCode::Add => Ok((a + b).into()),
                    OpCode::Subtract => Ok((a - b).into()),
                    OpCode::Less => Ok(Value::Bool(a < b)),
                    OpCode::Greater => Ok(Value::Bool(a > b)),
                    OpCode::LessEqual => Ok(Value::Bool(a <= b)),
                    OpCode::GreaterEqual => Ok(Value::Bool(a >= b)),
                    OpCode::Equal => Ok(Value::Bool(a == b)),
                    OpCode::NotEqual => Ok(Value::Bool(a != b)),
                    _ => {
                        Err(EvaluationError::InvalidBinaryOp {
                            operation,
                            left: Value::Duration(a),
                            right: Value::Duration(b),
                        })
                    }
                }
            }

            (op, Value::Duration(a), Value::Number(b))
            => {
                let b = b.as_i64().unwrap() as i32;

                match op {
                    OpCode::Multiply => Ok((a * b).into()),
                    OpCode::Divide => Ok((a / b).into()),
                    _ => {
                        Err(EvaluationError::InvalidBinaryOp {
                            operation,
                            left: Value::Duration(a),
                            right: Value::Number(Number::from(b)),
                        })
                    }
                }
            }

            (op, Value::Number(a), Value::Duration(b))
            => {
                let a = a.as_i64().unwrap() as i32;

                match op {
                    OpCode::Multiply => Ok((b * a).into()),
                    _ => {
                        Err(EvaluationError::InvalidBinaryOp {
                            operation,
                            left: Value::Number(Number::from(a)),
                            right: Value::Duration(b),
                        })
                    }
                }
            }

            (op, Value::Duration(a), Value::DateTime(b)) => {
                match op {
                    OpCode::Add => match b {
                        DateTime::Local(b) => { Ok((b + a).into()) }
                        DateTime::Utc(b) => { Ok((b + a).into()) }
                    }
                    _ => {
                        Err(EvaluationError::InvalidBinaryOp {
                            operation,
                            left: Value::Duration(a),
                            right: Value::DateTime(b),
                        })
                    }
                }
            }

            (op, Value::DateTime(a), Value::Duration(b))
            => {
                match op {
                    OpCode::Add => match a {
                        DateTime::Local(a) => { Ok((a + b).into()) }
                        DateTime::Utc(a) => { Ok((a + b).into()) }
                    }
                    OpCode::Subtract => match a {
                        DateTime::Local(a) => { Ok((a - b).into()) }
                        DateTime::Utc(a) => { Ok((a - b).into()) }
                    }
                    _ => {
                        Err(EvaluationError::InvalidBinaryOp {
                            operation,
                            left: Value::DateTime(a),
                            right: Value::Duration(b),
                        })
                    }
                }
            }

            (op, Value::Date(a), Value::Date(b))
            => {
                match op {
                    OpCode::Subtract => Ok((a - b).into()),
                    OpCode::Less => Ok(Value::Bool(a < b)),
                    OpCode::Greater => Ok(Value::Bool(a > b)),
                    OpCode::LessEqual => Ok(Value::Bool(a <= b)),
                    OpCode::GreaterEqual => Ok(Value::Bool(a >= b)),
                    OpCode::Equal => Ok(Value::Bool(a == b)),
                    OpCode::NotEqual => Ok(Value::Bool(a != b)),
                    _ => {
                        Err(EvaluationError::InvalidBinaryOp {
                            operation,
                            left: Value::Date(a),
                            right: Value::Date(b),
                        })
                    }
                }
            }

            (op, Value::Duration(a), Value::Date(b))
            => {
                match op {
                    OpCode::Add => Ok((b + a).into()),
                    _ => {
                        Err(EvaluationError::InvalidBinaryOp {
                            operation,
                            left: Value::Duration(a),
                            right: Value::Date(b),
                        })
                    }
                }
            }

            (op, Value::Date(a), Value::Duration(b))
            => {
                match op {
                    OpCode::Add => Ok((a + b).into()),
                    OpCode::Subtract => Ok((a - b).into()),
                    _ => {
                        Err(EvaluationError::InvalidBinaryOp {
                            operation,
                            left: Value::Date(a),
                            right: Value::Duration(b),
                        })
                    }
                }
            }

            (op, Value::Time(a), Value::Time(b))
            => {
                match op {
                    OpCode::Subtract => Ok((a - b).into()),
                    OpCode::Less => Ok(Value::Bool(a < b)),
                    OpCode::Greater => Ok(Value::Bool(a > b)),
                    OpCode::LessEqual => Ok(Value::Bool(a <= b)),
                    OpCode::GreaterEqual => Ok(Value::Bool(a >= b)),
                    OpCode::Equal => Ok(Value::Bool(a == b)),
                    OpCode::NotEqual => Ok(Value::Bool(a != b)),
                    _ => {
                        Err(EvaluationError::InvalidBinaryOp {
                            operation,
                            left: Value::Time(a),
                            right: Value::Time(b),
                        })
                    }
                }
            }

            (op, Value::Duration(a), Value::Time(b))
            => {
                match op {
                    OpCode::Add => Ok((b + a).into()),
                    _ => {
                        Err(EvaluationError::InvalidBinaryOp {
                            operation,
                            left: Value::Duration(a),
                            right: Value::Time(b),
                        })
                    }
                }
            }

            (op, Value::Time(a), Value::Duration(b))
            => {
                match op {
                    OpCode::Add => Ok((a + b).into()),
                    OpCode::Subtract => Ok((a - b).into()),
                    _ => {
                        Err(EvaluationError::InvalidBinaryOp {
                            operation,
                            left: Value::Time(a),
                            right: Value::Duration(b),
                        })
                    }
                }
            }

            (op, Value::SemVer(a), Value::SemVer(b)) => {
                Self::apply_semver_op(op, a, b)
            }

            (op, Value::SemVer(a), Value::String(b)) => {
                use anyhow::Context;
                let b = Version::parse(&b).with_context(|| format!("Not a valid version string {}", b))?;
                Self::apply_semver_op(op, a, b)
            }

            (op, Value::String(a), Value::SemVer(b)) => {
                use anyhow::Context;
                let a = Version::parse(&a).with_context(|| format!("Not a valid version string {}", a))?;
                Self::apply_semver_op(op, a, b)
            }

            (op, Value::String(a), Value::String(b)) => {
                match op {
                    OpCode::Add => Ok((format!("{}{}", a, b)).into()),
                    OpCode::Less => Ok(Value::Bool(a < b)),
                    OpCode::Greater => Ok(Value::Bool(a > b)),
                    OpCode::LessEqual => Ok(Value::Bool(a <= b)),
                    OpCode::GreaterEqual => Ok(Value::Bool(a >= b)),
                    OpCode::Equal => Ok(Value::Bool(a == b)),
                    OpCode::NotEqual => Ok(Value::Bool(a != b)),
                    OpCode::In => Ok(Value::Bool(b.contains(&a))),
                    _ => {
                        Err(EvaluationError::InvalidBinaryOp {
                            operation,
                            left: Value::String(a),
                            right: Value::String(b),
                        })
                    }
                }
            }

            (OpCode::In, left, Value::Array(v)) => Ok(Value::Bool(v.contains(&left))),

            // string operators with null
            (OpCode::Add, Value::Null, Value::String(b)) => Ok(Value::String(b)),
            (OpCode::Add, Value::String(a), Value::Null) => Ok(Value::String(a)),
            (OpCode::Add, Value::Null, Value::Null) => Ok(Value::Null),
            (OpCode::In, Value::String(_a), Value::Null) => Ok(Value::Bool(false)),
            (OpCode::In, Value::Null, Value::String(_b)) => Ok(Value::Bool(false)),
            (OpCode::In, Value::Null, Value::Null) => Ok(Value::Bool(false)),
            (OpCode::Equal, Value::String(_a), Value::Null) => Ok(Value::Bool(false)),
            (OpCode::Equal, Value::Null, Value::String(_b)) => Ok(Value::Bool(false)),
            (OpCode::Equal, Value::Null, Value::Null) => Ok(Value::Bool(true)),
            (OpCode::NotEqual, Value::String(_a), Value::Null) => Ok(Value::Bool(true)),
            (OpCode::NotEqual, Value::Null, Value::String(_b)) => Ok(Value::Bool(true)),
            (OpCode::NotEqual, Value::Null, Value::Null) => Ok(Value::Bool(false)),
            (operation, left, right) => Err(EvaluationError::InvalidBinaryOp {
                operation,
                left,
                right,
            }),
        }
    }

    fn apply_semver_op(op: OpCode, a: Version, b: Version) -> Result<Value> {
        match op {
            OpCode::Less => Ok(Value::Bool(a < b)),
            OpCode::Greater => Ok(Value::Bool(a > b)),
            OpCode::LessEqual => Ok(Value::Bool(a <= b)),
            OpCode::GreaterEqual => Ok(Value::Bool(a >= b)),
            OpCode::Equal => Ok(Value::Bool(a == b)),
            OpCode::NotEqual => Ok(Value::Bool(a != b)),
            _ => {
                Err(EvaluationError::InvalidBinaryOp {
                    operation: op,
                    left: Value::SemVer(a),
                    right: Value::SemVer(b),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use value::Value;

    use super::*;

    #[test]
    fn test_literal() {
        let expected: Value = 1.0.into();
        assert_eq!(Evaluator::new().eval("1").unwrap(), expected);
    }

    #[test]
    fn test_binary_expression_addition() {
        let expected: Value = 3.0.into();
        assert_eq!(Evaluator::new().eval("1 + 2").unwrap(), expected);
    }

    #[test]
    fn test_binary_expression_multiplication() {
        let expected: Value = 6.0.into();
        assert_eq!(Evaluator::new().eval("2 * 3").unwrap(), expected);
    }

    #[test]
    fn test_precedence() {
        let expected: Value = 14.0.into();
        assert_eq!(Evaluator::new().eval("2 + 3 * 4").unwrap(), expected);
    }

    #[test]
    fn test_parenthesis() {
        let expected: Value = 20.0.into();
        assert_eq!(Evaluator::new().eval("(2 + 3) * 4").unwrap(), expected);
    }

    #[test]
    fn test_string_concat() {
        assert_eq!(
            Evaluator::new().eval("'Hello ' + 'World'").unwrap(),
            Value::String("Hello World".to_string())
        );
    }

    #[test]
    fn test_string_ne() {
        assert_eq!(
            Evaluator::new().eval("'a' != 'b'").unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn test_simple_negation() {
        assert_eq!(
            Evaluator::new().eval("!('a' != 'b')").unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_complex_negation() {
        let context = json!({"a": {"b": false}});
        assert_eq!(
            Evaluator::new().eval_in_context("!a.b", context).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn test_double_negation() {
        assert_eq!(
            Evaluator::new().eval("!!'a'").unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn test_string_negation() {
        assert_eq!(
            Evaluator::new().eval("!'a'").unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_true_comparison() {
        assert_eq!(Evaluator::new().eval("2 > 1").unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_false_comparison() {
        assert_eq!(Evaluator::new().eval("2 <= 1").unwrap(), Value::Bool(false));
    }

    #[test]
    fn test_boolean_logic() {
        assert_eq!(
            Evaluator::new()
                .eval("'foo' && 6 >= 6 && 0 + 1 && true")
                .unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn test_identifier() {
        let context = json!({"a": 1.0});
        let expected: Value = 1.0.into();
        assert_eq!(
            Evaluator::new().eval_in_context("a", context).unwrap(),
            expected
        );
    }

    #[test]
    fn test_identifier_chain() {
        let context = json!({"a": {"b": 2.0}});
        let expected: Value = 2.0.into();
        assert_eq!(
            Evaluator::new().eval_in_context("a.b", context).unwrap(),
            expected
        );
    }

    #[test]
    fn test_identifier_default_value() {
        let context = json!({"a": {"b": 2.0}});
        let expected: Value = 3.0.into();
        assert_eq!(
            Evaluator::new().eval_in_context("a.get(c, 3.0)", context).unwrap(),
            expected
        );
    }

    #[test]
    fn test_missing_identifier() {
        let context = json!({"a": {"b": 2.0}});
        assert_eq!(
            Evaluator::new().eval_in_context("c", context).unwrap(),
            Value::Null
        );
    }

    #[test]
    fn test_missing_identifier_chain() {
        let context = json!({"a": {"b": 2.0, "c": {"d": 1.0}}});
        assert_eq!(
            Evaluator::new().eval_in_context("a.c1.d", context).unwrap(),
            Value::Null
        );
    }

    #[test]
    fn test_context_filter_arrays() {
        let context = json!({
            "foo": {
                "bar": [
                    {"tek": "hello"},
                    {"tek": "baz"},
                    {"tok": "baz"},
                ]
            }
        });

        let expected: Value = json!([{"tek": "baz"}]).into();
        assert_eq!(
            Evaluator::new()
                .eval_in_context("foo.bar[.tek == 'baz']", &context)
                .unwrap(),
            expected
        );
    }

    #[test]
    fn test_context_array_index() {
        let context = json!({
            "foo": {
                "bar": [
                    {"tek": "hello"},
                    {"tek": "baz"},
                    {"tok": "baz"},
                ]
            }
        });
        assert_eq!(
            Evaluator::new()
                .eval_in_context("foo.bar[1].tek", context)
                .unwrap(),
            Value::String("baz".to_string())
        );
    }

    #[test]
    fn test_context_array_oob_index() {
        let context = json!({
            "foo": {
                "bar": [
                    {"tek": "hello"},
                    {"tek": "baz"},
                    {"tok": "baz"},
                ]
            }
        });
        assert_eq!(
            Evaluator::new()
                .eval_in_context("foo.bar[4].tek", context)
                .unwrap(),
            Value::Null
        );
    }

    #[test]
    fn test_object_expression_properties() {
        let context = json!({"foo": {"baz": {"bar": "tek"}}});
        assert_eq!(
            Evaluator::new()
                .eval_in_context("foo['ba' + 'z'].bar", &context)
                .unwrap(),
            Value::String("tek".to_string())
        );
    }

    #[test]
    fn test_divfloor() {
        let expected: Value = 3.0.into();
        assert_eq!(Evaluator::new().eval("7 // 2").unwrap(), expected);
    }

    #[test]
    fn test_empty_object_literal() {
        let expected: Value = BTreeMap::new().into();
        assert_eq!(Evaluator::new().eval("{}").unwrap(), expected);
    }

    #[test]
    fn test_object_literal_strings() {
        let expected: Value = json!({"foo": {"bar": "tek"}}).into();
        assert_eq!(
            Evaluator::new().eval("{'foo': {'bar': 'tek'}}").unwrap(),
            expected
        );
    }

    #[test]
    fn test_object_literal_identifiers() {
        let expected: Value = json!({"foo": {"bar": "tek"}}).into();
        assert_eq!(
            Evaluator::new().eval("{foo: {bar: 'tek'}}").unwrap(),
            expected
        );
    }

    #[test]
    fn test_object_literal_properties() {
        assert_eq!(
            Evaluator::new().eval("{foo: 'bar'}.foo").unwrap(),
            Value::String("bar".to_string())
        );
    }

    #[test]
    fn test_array_literal() {
        let expected: Value = json!(["foo", 3.0]).into();
        assert_eq!(
            Evaluator::new().eval("['foo', 1+2]").unwrap(),
            expected
        );
    }

    #[test]
    fn test_array_literal_indexing() {
        let expected: Value = 2.0.into();
        assert_eq!(Evaluator::new().eval("[1, 2, 3][1]").unwrap(), expected);
    }

    #[test]
    fn test_in_operator_string() {
        assert_eq!(
            Evaluator::new().eval("'bar' in 'foobartek'").unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            Evaluator::new().eval("'baz' in 'foobartek'").unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_in_operator_array() {
        assert_eq!(
            Evaluator::new()
                .eval("'bar' in ['foo', 'bar', 'tek']")
                .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            Evaluator::new()
                .eval("'baz' in ['foo', 'bar', 'tek']")
                .unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn test_conditional_expression() {
        let expected: Value = (1f64).into();
        assert_eq!(
            Evaluator::new().eval("'foo' ? 1 : 2").unwrap(),
            expected
        );

        let expected: Value = (2f64).into();
        assert_eq!(Evaluator::new().eval("'' ? 1 : 2").unwrap(), expected);
    }

    #[test]
    fn test_arbitrary_whitespace() {
        let expected: Value = (20.0).into();
        assert_eq!(
            Evaluator::new().eval("(\t2\n+\n3) *\n4\n\r\n").unwrap(),
            expected
        );
    }

    #[test]
    fn test_non_integer() {
        let expected: Value = (4.5).into();
        assert_eq!(Evaluator::new().eval("1.5 * 3.0").unwrap(), expected);
    }

    #[test]
    fn test_string_literal() {
        assert_eq!(
            Evaluator::new().eval("'hello world'").unwrap(),
            Value::String("hello world".to_string())
        );
        assert_eq!(
            Evaluator::new().eval("\"hello world\"").unwrap(),
            Value::String("hello world".to_string())
        );
    }

    #[test]
    fn test_string_escapes() {
        assert_eq!(Evaluator::new().eval("'a\\'b'").unwrap(), Value::String("a'b".to_string()));
        assert_eq!(Evaluator::new().eval("\"a\\\"b\"").unwrap(), Value::String("a\"b".to_string()));
    }

    #[test]
    // Test a very simple transform that applies to_lowercase to a string
    fn test_simple_transform() {
        let evaluator = Evaluator::new().with_transform("lower", |v: Option<&[Value]>| {
            let s = v
                .expect("Must be valid argument")
                .get(0)
                .expect("There should be one argument!")
                .as_str()
                .expect("Should be a string!");
            Ok(Value::String(s.to_lowercase()))
        });
        assert_eq!(evaluator.eval("'T_T'.lower()").unwrap(), Value::String("t_t".to_string()));
    }

    #[test]
    // Test returning an UnknownTransform error if a transform is unknown
    fn test_missing_transform() {
        let err = Evaluator::new().eval("'hello'.world()").unwrap_err();
        if let EvaluationError::UnknownTransform(transform) = err {
            assert_eq!(transform, "world")
        } else {
            panic!("Should have thrown an unknown transform error")
        }
    }

    #[test]
    fn test_add_multiple_transforms() {
        let evaluator = Evaluator::new()
            .with_transform("sqrt", |v: Option<&[Value]>| {
                let num = v
                    .expect("Must be valid argument")
                    .first()
                    .expect("There should be one argument!")
                    .as_f64()
                    .expect("Should be a valid number!");
                Ok((num.sqrt() as u64).into())
            })
            .with_transform("square", |v: Option<&[Value]>| {
                let num = v
                    .expect("Must be valid argument")
                    .first()
                    .expect("There should be one argument!")
                    .as_f64()
                    .expect("Should be a valid number!");
                Ok(((num as u64).pow(2)).into())
            });

        let expected: Value = (16).into();
        assert_eq!(evaluator.eval("4.square()").unwrap(), expected);

        let expected: Value = (2).into();
        assert_eq!(evaluator.eval("4.sqrt()").unwrap(), expected);

        let expected: Value = (4).into();
        assert_eq!(evaluator.eval("4.square().sqrt()").unwrap(), expected);
    }

    #[test]
    fn test_transform_with_argument() {
        let evaluator = Evaluator::new().with_transform("split", |args: Option<&[Value]>| {
            let s = args
                .expect("Must be valid argument")
                .first()
                .expect("Should be a first argument!")
                .as_str()
                .expect("Should be a string!");
            let c = args
                .expect("Must be valid argument")
                .get(1)
                .expect("There should be a second argument!")
                .as_str()
                .expect("Should be a string");
            let res: Vec<&str> = s.split_terminator(c).collect();
            Ok(res.into())
        });

        let expected: Value = (vec!["John", "Doe"]).into();
        assert_eq!(
            evaluator.eval("'John Doe'.split(' ')").unwrap(),
            expected
        );
    }

    #[derive(Debug, thiserror::Error)]
    enum CustomError {
        #[error("Invalid argument in transform!")]
        InvalidArgument,
    }

    #[test]
    fn test_custom_error_message() {
        let evaluator = Evaluator::new().with_transform("error", |_: Option<&[Value]>| {
            Err(CustomError::InvalidArgument.into())
        });
        let res = evaluator.eval("1234.error()");
        assert!(res.is_err());
        if let EvaluationError::CustomError(e) = res.unwrap_err() {
            assert_eq!(e.to_string(), "Invalid argument in transform!")
        } else {
            panic!("Should have returned a Custom error!")
        }
    }

    #[test]
    fn test_filter_collections_many_returned() {
        let evaluator = Evaluator::new();
        let context = json!({
            "foo": [
                {"bobo": 50, "fofo": 100},
                {"bobo": 60, "baz": 90},
                {"bobo": 10, "bar": 83},
                {"bobo": 20, "yam": 12},
            ]
        });
        let exp = "foo[.bobo >= 50]";
        let expected: Value = json!([{"bobo": 50, "fofo": 100}, {"bobo": 60, "baz": 90}]).into();
        assert_eq!(
            evaluator.eval_in_context(exp, context).unwrap(),
            expected
        );
    }
}
