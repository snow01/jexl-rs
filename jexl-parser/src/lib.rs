/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

pub use lalrpop_util::ParseError;

pub use lalrpop_util::lexer::Token;

pub mod ast;
#[rustfmt::skip]
mod parser;

pub struct Parser {}

impl Parser {
    pub fn parse(input: &str) -> Result<ast::Expression, ParseError<usize, Token, &str>> {
        Ok(*parser::ExpressionParser::new().parse(input)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{Expression, OpCode, StdFunction, StringValue};

    use super::*;

    #[test]
    fn literal() {
        assert_eq!(Parser::parse("1"), Ok(Expression::Number(1.0)));
    }

    #[test]
    fn binary_expression() {
        assert_eq!(
            Parser::parse("1+2"),
            Ok(Expression::BinaryOperation {
                operation: OpCode::Add,
                left: Box::new(Expression::Number(1.0)),
                right: Box::new(Expression::Number(2.0)),
            }),
        );
    }

    #[test]
    fn binary_expression_whitespace() {
        assert_eq!(Parser::parse("1  +     2 "), Parser::parse("1+2"), );
    }

    // TODO: update this test
    #[test]
    fn transform_simple_no_args() {
        let exp = "'T_T'.lower()";
        let parsed = Parser::parse(exp).unwrap();
        assert_eq!(
            parsed,
            Expression::StdFunction(StdFunction::FuncLower(StringValue { value: Box::new(Expression::String("T_T".to_string())) }))
            //     name: "lower".to_string(),
            //     subject: Box::new(Expression::String("T_T".to_string())),
            //     args: None,
            // }
        );
    }

    // TODO: update this test
    #[test]
    fn transform_multiple_args() {
        let exp = "'John Doe'.lower().split(' ')";
        let parsed = Parser::parse(exp).unwrap();
        assert_eq!(
            parsed,
            Expression::StdFunction(StdFunction::FuncSplit {
                subject: StringValue {
                    value: Box::new(Expression::StdFunction(StdFunction::FuncLower(StringValue { value: Box::new(Expression::String("John Doe".to_string())) })))
                },
                with: StringValue {
                    value: Box::new(Expression::String(" ".to_string()))
                },
                num_splits: None,
            }),
            // Expression::CustomTransform {
            //     name: "split".to_string(),
            //     subject: Box::new(Expression::CustomTransform { name: "lower".to_string(), subject: Box::new(Expression::String("John Doe".to_string())), args: None }),
            //     args: Some(vec![Box::new(Expression::String(" ".to_string()))]),
            // }
        );
    }

    #[test]
    fn trasform_way_too_many_args() {
        let exp = "123456.math(12, 35, 100, 31, 90)";
        let parsed = Parser::parse(exp).unwrap();
        assert_eq!(
            parsed,
            Expression::CustomTransform {
                name: "math".to_string(),
                subject: Box::new(Expression::Number(123_456f64)),
                args: Some(vec![
                    Box::new(Expression::Number(12f64)),
                    Box::new(Expression::Number(35f64)),
                    Box::new(Expression::Number(100f64)),
                    Box::new(Expression::Number(31f64)),
                    Box::new(Expression::Number(90f64)),
                ]),
            }
        );
    }

    #[test]
    fn test_index_op_ident() {
        let exp = "foo[0]";
        let parsed = Parser::parse(exp).unwrap();
        assert_eq!(
            parsed,
            Expression::IndexOperation {
                subject: Box::new(Expression::Identifier("foo".to_string())),
                index: Box::new(Expression::Number(0f64)),
            }
        );
    }

    #[test]
    fn test_index_op_array_literal() {
        let exp = "[1, 2, 3][0]";
        let parsed = Parser::parse(exp).unwrap();
        assert_eq!(
            parsed,
            Expression::IndexOperation {
                subject: Box::new(Expression::Array(vec![
                    Box::new(Expression::Number(1f64)),
                    Box::new(Expression::Number(2f64)),
                    Box::new(Expression::Number(3f64)),
                ])),
                index: Box::new(Expression::Number(0f64)),
            }
        );
    }

    #[test]
    fn test_dot_op_ident() {
        let exp = "foo.bar";
        let parsed = Parser::parse(exp).unwrap();
        assert_eq!(
            parsed,
            Expression::DotOperation {
                subject: Box::new(Expression::Identifier("foo".to_string())),
                ident: "bar".to_string(),
                default_value: None,
            }
        );
    }

    #[test]
    fn test_dot_op_object_literal() {
        let exp = "{'foo': 1}.foo";
        let parsed = Parser::parse(exp).unwrap();
        assert_eq!(
            parsed,
            Expression::DotOperation {
                subject: Box::new(Expression::Object(vec![(
                    "foo".to_string(),
                    Box::new(Expression::Number(1f64))
                )])),
                ident: "foo".to_string(),
                default_value: None,
            }
        );
    }

    /// issue # 17
    #[test]
    fn test_string_literals_ne() {
        let exp = "'a' != 'b'";
        let parsed = Parser::parse(exp).unwrap();
        println!("Parsed: {:?}", parsed);
        assert_eq!(
            parsed,
            Expression::BinaryOperation {
                operation: OpCode::NotEqual,
                left: Box::new(Expression::String("a".to_string())),
                right: Box::new(Expression::String("b".to_string())),
            });
    }

    /// issue # 17
    #[test]
    fn test_negation_operation() {
        let exp = "! ('a' != 'b')";
        let parsed = Parser::parse(exp).unwrap();
        println!("Parsed: {:?}", parsed);
        assert_eq!(
            parsed,
            Expression::NegationOperation {
                expr: Box::new(Expression::BinaryOperation {
                    operation: OpCode::NotEqual,
                    left: Box::new(Expression::String("a".to_string())),
                    right: Box::new(Expression::String("b".to_string())),
                })
            });
    }

    #[test]
    fn test_double_negation() {
        let exp = "!! 'a'";
        let parsed = Parser::parse(exp).unwrap();
        println!("Parsed: {:?}", parsed);
        assert_eq!(
            parsed,
            Expression::NegationOperation {
                expr: Box::new(Expression::NegationOperation {
                    expr: Box::new(Expression::String("a".to_string())),
                })
            });
    }
}
