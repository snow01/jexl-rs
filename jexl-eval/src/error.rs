/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use jexl_parser::{ast::OpCode, ParseError, Token};

use crate::value::Value;

pub type Result<T, E = EvaluationError> = std::result::Result<T, E>;
#[derive(Debug, thiserror::Error)]
pub enum EvaluationError {
    #[error("Parsing error: {0}")]
    ParseError(String),

    #[error("Invalid binary operation, left: {left:?}, right: {right:?}, operation: {operation}")]
    InvalidBinaryOp {
        left: Value,
        right: Value,
        operation: OpCode,
    },
    #[error("Invalid negation operation, expr: {value:?}")]
    InvalidNegationOp {
        value: Value,
    },
    #[error("Unknown transform: {0}")]
    UnknownTransform(String),
    #[error("Duplicate object key: {0}")]
    DuplicateObjectKey(String),
    #[error("Invalid context provided")]
    InvalidContext,
    #[error("Invalid index type")]
    InvalidIndexType,
    #[error("Invalid json: {0}")]
    JSONError(#[from] serde_json::Error),
    #[error("Custom error: {0}")]
    CustomError(#[from] anyhow::Error),
    #[error("Invalid filter")]
    InvalidFilter,

    #[error("Invalid value type, expected: {expected}, got: {got}")]
    InvalidValueType {
        expected: String,
        got: String,
    }
}

impl <'a> From<ParseError<usize, Token<'a>, &'a str>> for EvaluationError {
    fn from(cause: ParseError<usize, Token<'a>, &'a str>) -> Self {
        EvaluationError::ParseError(cause.to_string())
    }
}
