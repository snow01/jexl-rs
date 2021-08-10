/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[derive(Debug, PartialEq)]
pub enum Expression {
    Number(f64),
    String(String),
    Boolean(bool),
    Array(Vec<Box<Expression>>),
    Object(Vec<(String, Box<Expression>)>),
    Identifier(String),

    NegationOperation {
        expr: Box<Expression>,
    },

    BinaryOperation {
        operation: OpCode,
        left: Box<Expression>,
        right: Box<Expression>,
    },

    CustomTransform {
        name: String,
        subject: Box<Expression>,
        args: Option<Vec<Box<Expression>>>,
    },

    StdFunction(StdFunction),

    CustomFunction {
        name: String,
        args: Option<Vec<Box<Expression>>>,
    },

    DotOperation {
        subject: Box<Expression>,
        ident: String,
        default_value: Option<Box<Expression>>,
    },

    IndexOperation {
        subject: Box<Expression>,
        index: Box<Expression>,
    },

    Conditional {
        left: Box<Expression>,
        truthy: Box<Expression>,
        falsy: Box<Expression>,
    },

    Filter {
        ident: String,
        op: OpCode,
        right: Box<Expression>,
    },
}

#[derive(Debug, PartialEq)]
pub enum StdFunction {
    // Array functions
    FuncAny(ArrayValue),
    FuncAll(ArrayValue),
    FuncMax(ArrayValue),
    FuncMin(ArrayValue),
    FuncSum(ArrayValue),

    // String or array functions
    FuncLen(Box<Expression>),
    FuncIsEmpty(Box<Expression>),

    // String transform functions
    FuncCapitalise(StringValue),
    FuncUpperCase(StringValue),
    FuncLowerCase(StringValue),
    FuncTitleCase(StringValue),
    FuncKebabCase(StringValue),
    FuncSnakeCase(StringValue),
    FuncSwapCase(StringValue),
    FuncTrainCase(StringValue),
    FuncPascalCase(StringValue),
    FuncShoutyKebabCase(StringValue),
    FuncShoutySnakeCase(StringValue),

    // String hashes
    FuncMD5(StringValue),
    FuncSHA256(StringValue),

    // String strip methods
    FuncStrip {
        subject: StringValue,
        with: Option<StringValue>,
    },
    FuncLStrip {
        subject: StringValue,
        with: Option<StringValue>,
    },
    FuncRStrip {
        subject: StringValue,
        with: Option<StringValue>,
    },

    // String checks
    FuncEndsWith {
        subject: StringValue,
        with: StringValue,
        start: Option<NumericValue>,
        end: Option<NumericValue>,
    },
    FuncStartsWith {
        subject: StringValue,
        with: StringValue,
        start: Option<NumericValue>,
        end: Option<NumericValue>,
    },

    // String boolean operations
    FuncIsDecimal(StringValue),
    FuncIsAlphaNum(StringValue),
    FuncIsAlpha(StringValue),
    FuncIsDigit(StringValue),
    FuncIsNumeric(StringValue),
    FuncIsSpace(StringValue),
    FuncIsIdentifier(StringValue),
    FuncIsLowerCase(StringValue),
    FuncIsUpperCase(StringValue),
    FuncIsTitleCase(StringValue),
    FuncIsCapitalise(StringValue),
    FuncIsKebabCase(StringValue),
    FuncIsSnakeCase(StringValue),
    FuncIsTrainCase(StringValue),
    FuncIsPascalCase(StringValue),
    FuncIsShoutyKebabCase(StringValue),
    FuncIsShoutySnakeCase(StringValue),

    // String splits
    FuncRSplit {
        subject: StringValue,
        with: StringValue,
        num_splits: Option<NumericValue>,
    },

    FuncSplit {
        subject: StringValue,
        with: StringValue,
        num_splits: Option<NumericValue>,
    },

    FuncWords(StringValue),

    // String find
    FuncIndex {
        subject: StringValue,
        search: StringValue,
        start: Option<NumericValue>,
        end: Option<NumericValue>,
    },
    FuncRIndex {
        subject: StringValue,
        search: StringValue,
        start: Option<NumericValue>,
        end: Option<NumericValue>,
    },
    FuncFind {
        subject: StringValue,
        search: StringValue,
        start: Option<NumericValue>,
        end: Option<NumericValue>,
    },
    FuncRFind {
        subject: StringValue,
        search: StringValue,
        start: Option<NumericValue>,
        end: Option<NumericValue>,
    },

    // Mathematical functions
    FuncAbs(NumericValue),
    FuncRound(NumericValue, Option<NumericValue>),

    FuncHex(NumericValue),
    FuncOct(NumericValue),

    // DateTime Operations
    FuncParseLocalDateTime(StringValue),
    FuncLocalDateTimeFromTimestampSecs(NumericValue),
    FuncLocalDateTimeFromTimestampMillis(NumericValue),
    FuncParseUtcDateTime(StringValue),
    FuncUtcDateTimeFromTimestampSecs(NumericValue),
    FuncUtcDateTimeFromTimestampMillis(NumericValue),

    // build Utc DateTime
    FuncUtcDateTime {
        y: Option<NumericValue>,
        m: Option<NumericValue>,
        d: Option<NumericValue>,
        h: Option<NumericValue>,
        mm: Option<NumericValue>,
        ss: Option<NumericValue>,
        ms: Option<NumericValue>,
        us: Option<NumericValue>,
    },

    // build Local DateTime
    FuncLocalDateTime {
        y: Option<NumericValue>,
        m: Option<NumericValue>,
        d: Option<NumericValue>,
        h: Option<NumericValue>,
        mm: Option<NumericValue>,
        ss: Option<NumericValue>,
        ms: Option<NumericValue>,
        us: Option<NumericValue>,
    },

    // build NaiveDate
    FuncDate {
        y: Option<NumericValue>,
        m: Option<NumericValue>,
        d: Option<NumericValue>,
    },

    // build NaiveTime
    FuncTime {
        h: Option<NumericValue>,
        mm: Option<NumericValue>,
        ss: Option<NumericValue>,
        ms: Option<NumericValue>,
        us: Option<NumericValue>,
    },

    // get individual fields year, month, day of month, day of year, week day, hour, minute, second, nanosecond
    FuncGetYearFromDate(DateLikeValue),
    FuncGetMonthFromDate(DateLikeValue),
    FuncGetDayOfMonthFromDate(DateLikeValue),
    FuncGetDayOfYearFromDate(DateLikeValue),
    FuncGetDayOfWeekFromDate(DateLikeValue),

    FuncGetHourFromTime(TimeLikeValue),
    FuncGetMinuteFromTime(TimeLikeValue),
    FuncGetSecondFromTime(TimeLikeValue),
    FuncGetMillisecondFromTime(TimeLikeValue),

    // get date(), time(), timestamp(), timestamp_in_millis(),
    FuncGetDateFromDateTime(DateTimeValue),
    FuncGetTimeFromDateTime(DateTimeValue),
    FuncGetTimestampFromDateTime(DateTimeValue),
    FuncGetTimestampMillisFromDateTime(DateTimeValue),

    // duration
    FuncDuration {
        days: Option<NumericValue>,
        hours: Option<NumericValue>,
        minutes: Option<NumericValue>,
        seconds: Option<NumericValue>,
        milliseconds: Option<NumericValue>,
        microseconds: Option<NumericValue>,
        weeks: Option<NumericValue>,
    },

    FuncParseDuration(StringValue),

    // from duration - get weeks, days, hours, minutes, seconds, milliseconds, microseconds,
    FuncGetWeeksFromDuration(DurationValue),
    FuncGetDaysFromDuration(DurationValue),
    FuncGetHoursFromDuration(DurationValue),
    FuncGetMinutesFromDuration(DurationValue),
    FuncGetSecondsFromDuration(DurationValue),
    FuncGetMillisecondsFromDuration(DurationValue),
    FuncGetMicrosecondsFromDuration(DurationValue),

    FuncLocalNow(),
    FuncUtcNow(),

    // Semantic Version Function
    FuncSemVersion(StringValue),
}

#[derive(Debug, PartialEq)]
pub struct NumericValue {
    pub value: Box<Expression>,
}

#[derive(Debug, PartialEq)]
pub struct StringValue {
    pub value: Box<Expression>,
}

#[derive(Debug, PartialEq)]
pub struct BooleanValue {
    pub value: Box<Expression>,
}

#[derive(Debug, PartialEq)]
pub struct ArrayValue {
    pub array: Box<Expression>,
}

#[derive(Debug, PartialEq)]
pub struct DateTimeValue {
    pub value: Box<Expression>,
}

#[derive(Debug, PartialEq)]
pub struct DateLikeValue {
    pub value: Box<Expression>,
}

#[derive(Debug, PartialEq)]
pub struct TimeLikeValue {
    pub value: Box<Expression>,
}

#[derive(Debug, PartialEq)]
pub struct DurationValue {
    pub value: Box<Expression>,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum OpCode {
    Add,
    Subtract,
    Multiply,
    Divide,
    FloorDivide,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    And,
    Or,
    Modulus,
    Exponent,
    In,
}

impl std::fmt::Display for OpCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                OpCode::Add => "Add",
                OpCode::Subtract => "Subtract",
                OpCode::Multiply => "Multiply",
                OpCode::Divide => "Divide",
                OpCode::FloorDivide => "Floor division",
                OpCode::Less => "Less than",
                OpCode::LessEqual => "Less than or equal to",
                OpCode::Greater => "Greater than",
                OpCode::GreaterEqual => "Greater than or equal to",
                OpCode::Equal => "Equal",
                OpCode::NotEqual => "Not equal",
                OpCode::And => "Bitwise And",
                OpCode::Or => "Bitwise Or",
                OpCode::Modulus => "Modulus",
                OpCode::Exponent => "Exponent",
                OpCode::In => "In",
            }
        )
    }
}
