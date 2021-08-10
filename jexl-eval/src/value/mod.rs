use std::collections::BTreeMap;
use std::fmt;
use std::fmt::Formatter;

use chrono::{Datelike, NaiveDate, NaiveTime, Timelike, Weekday};
use serde::Serialize;
pub use serde_json::Number;

pub use error::Error;
pub use error::Result;
use ser::Serializer;

macro_rules! tri {
    ($e:expr) => {
        match $e {
            std::result::Result::Ok(val) => val,
            std::result::Result::Err(err) => return std::result::Result::Err(err),
        }
    };
    ($e:expr,) => {
        tri!($e)
    };
}

mod ser;
mod de;
mod error;
mod from;
mod index;
mod partial_eq;

#[derive(Clone, Eq, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
    DateTime(DateTime),
    Date(chrono::NaiveDate),
    Time(chrono::NaiveTime),
    Duration(chrono::Duration),
    SemVer(semver::Version),
}

#[derive(Clone, Eq, PartialEq, Serialize)]
pub enum DateTime {
    Local(chrono::DateTime<chrono::Local>),
    Utc(chrono::DateTime<chrono::Utc>),
}

pub enum DateLike {
    LocalDateTime(chrono::DateTime<chrono::Local>),
    UtcDateTime(chrono::DateTime<chrono::Utc>),
    Date(chrono::NaiveDate),
}

pub enum TimeLike {
    LocalDateTime(chrono::DateTime<chrono::Local>),
    UtcDateTime(chrono::DateTime<chrono::Utc>),
    Time(chrono::NaiveTime),
}

impl DateTime {
    pub fn timestamp(&self) -> i64 {
        match self {
            DateTime::Local(v) => { v.timestamp() }
            DateTime::Utc(v) => { v.timestamp() }
        }
    }

    pub fn timestamp_millis(&self) -> i64 {
        match self {
            DateTime::Local(v) => { v.timestamp_millis() }
            DateTime::Utc(v) => { v.timestamp_millis() }
        }
    }

    pub fn date(&self) -> NaiveDate {
        match self {
            DateTime::Local(v) => { v.date().naive_local() }
            DateTime::Utc(v) => { v.date().naive_local() }
        }
    }

    pub fn time(&self) -> NaiveTime {
        match self {
            DateTime::Local(v) => { v.time() }
            DateTime::Utc(v) => { v.time() }
        }
    }
}

impl DateLike {
    pub fn year(&self) -> i32 {
        match self {
            Self::LocalDateTime(v) => { v.year() }
            Self::UtcDateTime(v) => { v.year() }
            Self::Date(v) => { v.year() }
        }
    }

    pub fn month(&self) -> u32 {
        match self {
            Self::LocalDateTime(v) => { v.month() }
            Self::UtcDateTime(v) => { v.month() }
            Self::Date(v) => { v.month() }
        }
    }

    pub fn day(&self) -> u32 {
        match self {
            Self::LocalDateTime(v) => { v.day() }
            Self::UtcDateTime(v) => { v.day() }
            Self::Date(v) => { v.day() }
        }
    }

    pub fn ordinal(&self) -> u32 {
        match self {
            Self::LocalDateTime(v) => { v.ordinal() }
            Self::UtcDateTime(v) => { v.ordinal() }
            Self::Date(v) => { v.ordinal() }
        }
    }

    pub fn weekday(&self) -> Weekday {
        match self {
            Self::LocalDateTime(v) => { v.weekday() }
            Self::UtcDateTime(v) => { v.weekday() }
            Self::Date(v) => { v.weekday() }
        }
    }
}

impl TimeLike {
    pub fn hour(&self) -> u32 {
        match self {
            Self::LocalDateTime(v) => { v.hour() }
            Self::UtcDateTime(v) => { v.hour() }
            Self::Time(v) => { v.hour() }
        }
    }

    pub fn minute(&self) -> u32 {
        match self {
            Self::LocalDateTime(v) => { v.minute() }
            Self::UtcDateTime(v) => { v.minute() }
            Self::Time(v) => { v.minute() }
        }
    }

    pub fn second(&self) -> u32 {
        match self {
            Self::LocalDateTime(v) => { v.second() }
            Self::UtcDateTime(v) => { v.second() }
            Self::Time(v) => { v.second() }
        }
    }

    pub fn millisecond(&self) -> u32 {
        match self {
            Self::LocalDateTime(v) => { v.nanosecond() / 10 ^ 6 }
            Self::UtcDateTime(v) => { v.nanosecond() / 10 ^ 6 }
            Self::Time(v) => { v.nanosecond() / 10 ^ 6 }
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Null => formatter.debug_tuple("Null").finish(),
            Value::Bool(v) => formatter.debug_tuple("Bool").field(&v).finish(),
            Value::Number(ref v) => fmt::Debug::fmt(v, formatter),
            Value::String(ref v) => formatter.debug_tuple("String").field(v).finish(),
            Value::Array(ref v) => {
                formatter.write_str("Array(")?;
                fmt::Debug::fmt(v, formatter)?;
                formatter.write_str(")")
            }
            Value::Object(ref v) => {
                formatter.write_str("Object(")?;
                fmt::Debug::fmt(v, formatter)?;
                formatter.write_str(")")
            }
            Value::DateTime(v) => formatter.debug_tuple("DateTime").field(v).finish(),
            Value::Date(v) => formatter.debug_tuple("Date").field(v).finish(),
            Value::Time(v) => formatter.debug_tuple("Time").field(v).finish(),
            Value::Duration(v) => formatter.debug_tuple("Duration").field(v).finish(),
            Value::SemVer(v) => formatter.debug_tuple("SemVer").field(v).finish(),
        }
    }
}

impl fmt::Debug for DateTime {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            DateTime::Local(v) => formatter.debug_tuple("Local").field(&v).finish(),
            DateTime::Utc(v) => formatter.debug_tuple("Utc").field(&v).finish(),
        }
    }
}

impl Default for Value {
    fn default() -> Value {
        Value::Null
    }
}

// impl fmt::Display for Value {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         use std::io;
//
//         struct WriterFormatter<'a, 'b: 'a> {
//             inner: &'a mut fmt::Formatter<'b>,
//         }
//
//         impl<'a, 'b> io::Write for WriterFormatter<'a, 'b> {
//             fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
//                 // Safety: the serializer below only emits valid utf8 when using
//                 // the default formatter.
//                 let s = unsafe { std::str::from_utf8_unchecked(buf) };
//                 tri!(self.inner.write_str(s).map_err(io_error));
//                 Ok(buf.len())
//             }
//
//             fn flush(&mut self) -> io::Result<()> {
//                 Ok(())
//             }
//         }
//
//         fn io_error(_: fmt::Error) -> io::Error {
//             // Error value does not matter because Display impl just maps it
//             // back to fmt::Error.
//             io::Error::new(io::ErrorKind::Other, "fmt error")
//         }
//
//         let alternate = f.alternate();
//         let mut wr = WriterFormatter { inner: f };
//         // if alternate {
//         //     // {:#}
//         //     ser::to_writer_pretty(&mut wr, self).map_err(|_| fmt::Error)
//         // } else {
//         // {}
//         ser::to_writer(&mut wr, self).map_err(|_| fmt::Error)
//         // }
//     }
// }

fn parse_index(s: &str) -> Option<usize> {
    if s.starts_with('+') || (s.starts_with('0') && s.len() != 1) {
        return None;
    }
    s.parse().ok()
}

impl Value {
    pub fn get<I: index::Index>(&self, index: I) -> Option<&Value> {
        index.index_into(self)
    }

    pub fn get_mut<I: index::Index>(&mut self, index: I) -> Option<&mut Value> {
        index.index_into_mut(self)
    }

    pub fn is_object(&self) -> bool {
        self.as_object().is_some()
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, Value>> {
        match *self {
            Value::Object(ref map) => Some(map),
            _ => None,
        }
    }

    pub fn as_object_mut(&mut self) -> Option<&mut BTreeMap<String, Value>> {
        match *self {
            Value::Object(ref mut map) => Some(map),
            _ => None,
        }
    }

    pub fn is_array(&self) -> bool {
        self.as_array().is_some()
    }

    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match *self {
            Value::Array(ref array) => Some(&*array),
            _ => None,
        }
    }

    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Value>> {
        match *self {
            Value::Array(ref mut list) => Some(list),
            _ => None,
        }
    }

    pub fn is_string(&self) -> bool {
        self.as_str().is_some()
    }

    pub fn as_str(&self) -> Option<&str> {
        match *self {
            Value::String(ref s) => Some(s),
            _ => None,
        }
    }

    pub fn is_number(&self) -> bool {
        match *self {
            Value::Number(_) => true,
            _ => false,
        }
    }

    pub fn is_i64(&self) -> bool {
        match *self {
            Value::Number(ref n) => n.is_i64(),
            _ => false,
        }
    }

    pub fn is_u64(&self) -> bool {
        match *self {
            Value::Number(ref n) => n.is_u64(),
            _ => false,
        }
    }

    pub fn is_f64(&self) -> bool {
        match *self {
            Value::Number(ref n) => n.is_f64(),
            _ => false,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Value::Number(ref n) => n.as_i64(),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::Number(ref n) => n.as_u64(),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::Number(ref n) => n.as_f64(),
            _ => None,
        }
    }

    pub fn is_boolean(&self) -> bool {
        self.as_bool().is_some()
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        self.as_null().is_some()
    }

    pub fn as_null(&self) -> Option<()> {
        match *self {
            Value::Null => Some(()),
            _ => None,
        }
    }

    pub fn pointer(&self, pointer: &str) -> Option<&Value> {
        if pointer.is_empty() {
            return Some(self);
        }
        if !pointer.starts_with('/') {
            return None;
        }
        pointer
            .split('/')
            .skip(1)
            .map(|x| x.replace("~1", "/").replace("~0", "~"))
            .try_fold(self, |target, token| match target {
                Value::Object(map) => map.get(&token),
                Value::Array(list) => parse_index(&token).and_then(|x| list.get(x)),
                _ => None,
            })
    }

    pub fn pointer_mut(&mut self, pointer: &str) -> Option<&mut Value> {
        if pointer.is_empty() {
            return Some(self);
        }
        if !pointer.starts_with('/') {
            return None;
        }
        pointer
            .split('/')
            .skip(1)
            .map(|x| x.replace("~1", "/").replace("~0", "~"))
            .try_fold(self, |target, token| match target {
                Value::Object(map) => map.get_mut(&token),
                Value::Array(list) => parse_index(&token).and_then(move |x| list.get_mut(x)),
                _ => None,
            })
    }

    pub fn take(&mut self) -> Value {
        std::mem::replace(self, Value::Null)
    }
}

pub fn to_value<T>(value: T) -> Result<Value>
    where
        T: Serialize,
{
    value.serialize(Serializer)
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use chrono::{Local, NaiveDate, NaiveTime, Utc};

    use crate::value::to_value;

    #[derive(Debug, serde::Serialize)]
    struct A {
        a: String,
        dt_local: chrono::DateTime<Local>,
        dt_utc: chrono::DateTime<Utc>,
        d: NaiveDate,
        t: NaiveTime,
        // dr: Duration,
    }

    #[test]
    fn test_serialise() -> anyhow::Result<()> {
        let a = A {
            a: "abc".to_string(),
            dt_local: Local::now(),
            dt_utc: Utc::now(),
            d: Local::now().date().naive_local(),
            t: Local::now().time(),
            // dr: Duration::seconds(100)
        };

        let ser_value = to_value(&a).with_context(|| format!("Failed to serialise type to Value"))?;

        println!("Result: {:?}", ser_value);

        Ok(())
    }
}