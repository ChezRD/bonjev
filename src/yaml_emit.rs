//! Stream YAML events through libyaml. A mapping can repeat a key, which a
//! document tree cannot.

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr;
use std::slice;

use unsafe_libyaml::{
    yaml_document_end_event_initialize, yaml_document_start_event_initialize, yaml_emitter_delete,
    yaml_emitter_emit, yaml_emitter_flush, yaml_emitter_initialize, yaml_emitter_set_indent,
    yaml_emitter_set_output, yaml_emitter_set_unicode, yaml_emitter_set_width, yaml_emitter_t,
    yaml_encoding_t, yaml_event_t, yaml_mapping_end_event_initialize,
    yaml_mapping_start_event_initialize, yaml_mapping_style_t, yaml_scalar_event_initialize,
    yaml_scalar_style_t, yaml_stream_end_event_initialize, yaml_stream_start_event_initialize,
};

#[cfg(test)]
use unsafe_libyaml::{
    yaml_sequence_end_event_initialize, yaml_sequence_start_event_initialize, yaml_sequence_style_t,
};

#[cfg(test)]
use std::ffi::CStr;
#[cfg(test)]
use unsafe_libyaml::{
    YAML_LITERAL_SCALAR_STYLE, YAML_PLAIN_SCALAR_STYLE, YAML_SCALAR_EVENT, YAML_STREAM_END_EVENT,
    yaml_event_delete, yaml_parser_delete, yaml_parser_initialize, yaml_parser_parse,
    yaml_parser_set_input_string, yaml_parser_t,
};

macro_rules! emit {
    ($stream:expr, $init:expr) => {{
        let mut event = MaybeUninit::<yaml_event_t>::uninit();
        let event = event.as_mut_ptr();
        let status = $init(event);
        assert!(status.ok, "yaml event init failed");
        let status = unsafe { yaml_emitter_emit(&mut $stream.inner.emitter, event) };
        assert!(status.ok, "yaml emit failed");
    }};
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarStyle {
    Plain,
    Literal,
    Other,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scalar {
    pub text: String,
    pub style: ScalarStyle,
}

struct Inner {
    emitter: yaml_emitter_t,
    buf: Vec<u8>,
}

pub struct YamlStream {
    inner: Box<Inner>,
}

impl YamlStream {
    pub fn new() -> Self {
        let mut inner = Box::new(Inner {
            emitter: unsafe { MaybeUninit::<yaml_emitter_t>::zeroed().assume_init() },
            buf: Vec::new(),
        });
        unsafe {
            let ok = yaml_emitter_initialize(&mut inner.emitter);
            assert!(ok.ok, "yaml emitter init failed");
            yaml_emitter_set_unicode(&mut inner.emitter, true);
            yaml_emitter_set_width(&mut inner.emitter, -1);
            yaml_emitter_set_indent(&mut inner.emitter, 2);
            let buf_ptr = &mut inner.buf as *mut Vec<u8> as *mut c_void;
            yaml_emitter_set_output(
                &mut inner.emitter,
                write_handler,
                buf_ptr,
            );
        }
        let mut out = Self { inner };
        emit!(out, |event| unsafe {
            yaml_stream_start_event_initialize(event, yaml_encoding_t::YAML_UTF8_ENCODING)
        });
        emit!(out, |event| unsafe {
            yaml_document_start_event_initialize(
                event,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                true,
            )
        });
        out
    }

    pub fn mapping_start(&mut self) {
        emit!(self, |event| unsafe {
            yaml_mapping_start_event_initialize(
                event,
                ptr::null(),
                ptr::null(),
                true,
                yaml_mapping_style_t::YAML_BLOCK_MAPPING_STYLE,
            )
        });
    }

    pub fn mapping_end(&mut self) {
        emit!(self, |event| unsafe {
            yaml_mapping_end_event_initialize(event)
        });
    }

    pub fn key(&mut self, text: &str) {
        self.plain(text);
    }

    pub fn plain(&mut self, text: &str) {
        self.scalar(text, yaml_scalar_style_t::YAML_PLAIN_SCALAR_STYLE);
    }

    #[cfg(test)]
    pub fn sequence_start(&mut self) {
        emit!(self, |event| unsafe {
            yaml_sequence_start_event_initialize(
                event,
                ptr::null(),
                ptr::null(),
                true,
                yaml_sequence_style_t::YAML_BLOCK_SEQUENCE_STYLE,
            )
        });
    }

    #[cfg(test)]
    pub fn sequence_end(&mut self) {
        emit!(self, |event| unsafe {
            yaml_sequence_end_event_initialize(event)
        });
    }

    pub fn literal(&mut self, text: &str) {
        let owned;
        let value = if text.ends_with('\n') {
            text
        } else {
            owned = format!("{text}\n");
            owned.as_str()
        };
        self.scalar(value, yaml_scalar_style_t::YAML_LITERAL_SCALAR_STYLE);
    }

    fn scalar(&mut self, text: &str, style: yaml_scalar_style_t) {
        emit!(self, |event| unsafe {
            yaml_scalar_event_initialize(
                event,
                ptr::null(),
                ptr::null(),
                text.as_ptr(),
                text.len() as i32,
                true,
                true,
                style,
            )
        });
    }

    pub fn entry_literal(&mut self, key: &str, value: &str) {
        self.key(key);
        self.literal(value);
    }

    pub fn finish(mut self) -> String {
        emit!(self, |event| unsafe {
            yaml_document_end_event_initialize(event, true)
        });
        emit!(self, |event| unsafe {
            yaml_stream_end_event_initialize(event)
        });
        unsafe {
            let ok = yaml_emitter_flush(&mut self.inner.emitter);
            assert!(ok.ok, "yaml emitter flush failed");
        }
        let bytes = std::mem::take(&mut self.inner.buf);
        String::from_utf8(bytes).expect("yaml emitter wrote utf-8")
    }
}

unsafe fn write_handler(data: *mut c_void, buffer: *mut u8, size: u64) -> i32 {
    unsafe {
        let buf = &mut *(data as *mut Vec<u8>);
        buf.extend_from_slice(slice::from_raw_parts(buffer, size as usize));
    }
    1
}

impl Drop for YamlStream {
    fn drop(&mut self) {
        unsafe { yaml_emitter_delete(&mut self.inner.emitter) }
    }
}

/// Scalars in stream order. A repeated key stays repeated: a document tree would collapse it.
#[cfg(test)]
pub fn parse_scalars(yaml: &str) -> Result<Vec<Scalar>, String> {
    // SAFETY: the parser reads `yaml` only until `yaml_parser_delete`. Each event is deleted
    // before the next parse, and the parser is deleted on every exit.
    unsafe { parse_scalars_raw(yaml.as_bytes()) }
}

#[cfg(test)]
unsafe fn parse_scalars_raw(yaml: &[u8]) -> Result<Vec<Scalar>, String> {
    unsafe {
        let mut parser = MaybeUninit::<yaml_parser_t>::uninit();
        let parser = parser.as_mut_ptr();
        if yaml_parser_initialize(parser).fail {
            return Err("yaml parser init failed".to_string());
        }
        yaml_parser_set_input_string(parser, yaml.as_ptr(), yaml.len() as u64);
        let mut out = Vec::new();
        loop {
            let mut event = MaybeUninit::<yaml_event_t>::uninit();
            let event = event.as_mut_ptr();
            if yaml_parser_parse(parser, event).fail {
                let message = parser_problem(parser);
                yaml_parser_delete(parser);
                return Err(message);
            }
            if (*event).type_ == YAML_STREAM_END_EVENT {
                yaml_event_delete(event);
                yaml_parser_delete(parser);
                return Ok(out);
            }
            if (*event).type_ == YAML_SCALAR_EVENT {
                let raw = slice::from_raw_parts(
                    (*event).data.scalar.value,
                    (*event).data.scalar.length as usize,
                );
                let text = match String::from_utf8(raw.to_vec()) {
                    Ok(text) => text,
                    Err(_) => {
                        yaml_event_delete(event);
                        yaml_parser_delete(parser);
                        return Err("yaml scalar is not utf-8".to_string());
                    }
                };
                let style = (*event).data.scalar.style;
                let style = if style == YAML_LITERAL_SCALAR_STYLE {
                    ScalarStyle::Literal
                } else if style == YAML_PLAIN_SCALAR_STYLE {
                    ScalarStyle::Plain
                } else {
                    ScalarStyle::Other
                };
                out.push(Scalar { text, style });
            }
            yaml_event_delete(event);
        }
    }
}

#[cfg(test)]
unsafe fn parser_problem(parser: *mut yaml_parser_t) -> String {
    unsafe {
        let problem = (&*parser).problem;
        if problem.is_null() {
            return "yaml parse failed".to_string();
        }
        CStr::from_ptr(problem.cast())
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_emits_valid_yaml() {
        let mut doc = YamlStream::new();
        doc.mapping_start();
        doc.key("items");
        doc.sequence_start();
        doc.plain("first");
        doc.plain("second");
        doc.sequence_end();
        doc.mapping_end();
        let yaml = doc.finish();
        let scalars = parse_scalars(&yaml).unwrap();
        assert_eq!(
            scalars,
            [
                Scalar {
                    text: "items".to_string(),
                    style: ScalarStyle::Plain
                },
                Scalar {
                    text: "first".to_string(),
                    style: ScalarStyle::Plain
                },
                Scalar {
                    text: "second".to_string(),
                    style: ScalarStyle::Plain
                },
            ]
        );
    }
}
