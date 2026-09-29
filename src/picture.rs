use anyhow::{Context, Result};
use base64::Engine;
use image::ImageReader;
use std::io::Cursor;

pub struct Rgb {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

pub fn decode_bytes(bytes: &[u8]) -> Result<Rgb> {
    let img = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("image format")?
        .decode()
        .context("decode image")?;
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    if width == 0 || height == 0 {
        anyhow::bail!("image has no pixels");
    }
    Ok(Rgb {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

pub fn decode_base64(text: &str) -> Result<Rgb> {
    let payload = base64_payload(text);
    let cleaned = normalize_b64(payload);
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(cleaned.as_bytes())
        .context("image base64")?;
    decode_bytes(&bytes)
}

fn base64_payload(text: &str) -> &str {
    let text = text.trim();
    if let Some(rest) = text.strip_prefix("data:")
        && let Some((_, payload)) = rest.split_once(',')
    {
        return payload.trim();
    }
    text
}

fn normalize_b64(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 3);
    for c in text.chars() {
        if !c.is_whitespace() {
            out.push(c);
        }
    }
    while !out.len().is_multiple_of(4) {
        out.push('=');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageFormat, RgbImage};

    fn tiny_png() -> Vec<u8> {
        let mut img = RgbImage::new(2, 1);
        img.put_pixel(0, 0, image::Rgb([1, 2, 3]));
        img.put_pixel(1, 0, image::Rgb([4, 5, 6]));
        let mut buf = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut buf, ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    }

    #[test]
    fn png_decodes_to_rgb() {
        let rgb = decode_bytes(&tiny_png()).unwrap();
        assert_eq!((rgb.width, rgb.height), (2, 1));
        assert_eq!(rgb.pixels, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn data_url_decodes() {
        let b64 = base64::engine::general_purpose::STANDARD.encode(tiny_png());
        let rgb = decode_base64(&format!("data:image/png;base64,{b64}")).unwrap();
        assert_eq!(rgb.pixels, vec![1, 2, 3, 4, 5, 6]);
    }
}
