use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

fn timestamp_seconds(value: &str) -> Result<f64, String> {
    let fields: Vec<&str> = value.trim().split([':', ',', '.']).collect();
    if fields.len() != 4 {
        return Err(format!("无效时间戳: {value}"));
    }
    let numbers: Vec<f64> = fields
        .iter()
        .map(|part| {
            part.parse::<f64>()
                .map_err(|_| format!("无效时间戳: {value}"))
        })
        .collect::<Result<_, _>>()?;
    Ok(numbers[0] * 3600.0 + numbers[1] * 60.0 + numbers[2] + numbers[3] / 1000.0)
}

pub fn parse_srt(input: &str) -> Result<Vec<Segment>, String> {
    let normalized = input.replace("\r\n", "\n").replace('\r', "\n");
    let mut segments = Vec::new();
    for block in normalized.split("\n\n") {
        let lines: Vec<&str> = block
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect();
        let Some(index) = lines.iter().position(|line| line.contains("-->")) else {
            continue;
        };
        let (start, end) = lines[index].split_once("-->").ok_or("字幕时间轴无效")?;
        let text = lines[index + 1..].join("\n");
        if text.is_empty() {
            continue;
        }
        segments.push(Segment {
            start: timestamp_seconds(start)?,
            end: timestamp_seconds(end)?,
            text,
        });
    }
    if segments.is_empty() {
        return Err("引擎未生成可用字幕，请检查音频内容和模型。".into());
    }
    Ok(segments)
}

pub fn whole_transcript(segments: Vec<Segment>) -> Result<Vec<Segment>, String> {
    let end = segments
        .iter()
        .map(|segment| segment.end)
        .fold(0.0, f64::max);
    let mut text = String::new();
    for segment in &segments {
        let cleaned = segment
            .text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let part = cleaned.as_str();
        if part.is_empty() {
            continue;
        }
        if text.chars().last().is_some_and(|ch| ch.is_ascii())
            && part
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_alphanumeric())
        {
            text.push(' ');
        }
        text.push_str(part);
    }
    if end <= 0.0 || text.is_empty() {
        return Err("引擎未生成可用转录文本。".into());
    }
    Ok(vec![Segment {
        start: 0.0,
        end,
        text,
    }])
}

fn emit_aligned_line(tokens: &mut Vec<Segment>, lines: &mut Vec<Segment>) {
    if tokens.is_empty() {
        return;
    }
    let words = std::mem::take(tokens);
    let start = words[0].start;
    let end = words.last().unwrap().end;
    let separator = if words
        .iter()
        .any(|word| word.text.chars().any(|ch| ch.is_ascii_alphabetic()))
    {
        " "
    } else {
        ""
    };
    let text = words
        .iter()
        .map(|word| word.text.as_str())
        .collect::<Vec<_>>()
        .join(separator);
    if end > start && !text.trim().is_empty() {
        lines.push(Segment { start, end, text });
    }
}

pub fn split_aligned_srt(input: &str) -> Result<Vec<Segment>, String> {
    let cues = parse_srt(input)?;
    if cues
        .iter()
        .any(|cue| cue.text.chars().count() > 40 && cue.end - cue.start > 4.0)
    {
        return Err("对齐器未生成字级时间轴，请检查 Qwen3 对齐模型。".into());
    }
    let mut lines = Vec::new();
    let mut current: Vec<Segment> = Vec::new();
    let punctuation = "，。？！；：…—、·.,!?;";
    for cue in cues {
        if cue.text.chars().all(|ch| punctuation.contains(ch)) {
            emit_aligned_line(&mut current, &mut lines);
            continue;
        }
        if let Some(previous) = current.last() {
            let has_latin = current
                .iter()
                .any(|item| item.text.chars().any(|ch| ch.is_ascii_alphabetic()));
            let long_enough = if has_latin {
                current.len() >= 3
            } else {
                current
                    .iter()
                    .map(|item| item.text.chars().count())
                    .sum::<usize>()
                    >= 5
            };
            if long_enough && cue.start - previous.end >= 0.35 {
                emit_aligned_line(&mut current, &mut lines);
            }
        }
        current.push(cue);
        let has_latin = current
            .iter()
            .any(|item| item.text.chars().any(|ch| ch.is_ascii_alphabetic()));
        let over_limit = if has_latin {
            current.len() > 8
        } else {
            current
                .iter()
                .map(|item| item.text.chars().count())
                .sum::<usize>()
                > 20
        };
        if over_limit {
            let mut best = None;
            let mut best_gap = 0.15;
            for index in (current.len() / 3).max(1)..current.len() {
                let gap = current[index].start - current[index - 1].end;
                if gap >= best_gap {
                    best = Some(index);
                    best_gap = gap;
                }
            }
            if let Some(index) = best {
                let tail = current.split_off(index);
                emit_aligned_line(&mut current, &mut lines);
                current = tail;
            } else {
                emit_aligned_line(&mut current, &mut lines);
            }
        }
    }
    emit_aligned_line(&mut current, &mut lines);
    let mut merged: Vec<Segment> = Vec::new();
    for line in lines {
        if line.text.chars().count() == 1
            && !line.text.chars().any(|ch| ch.is_ascii_alphabetic())
            && merged
                .last()
                .is_some_and(|previous| line.start - previous.end <= 0.8)
        {
            let previous = merged.last_mut().unwrap();
            previous.text.push_str(&line.text);
            previous.end = line.end;
        } else {
            merged.push(line);
        }
    }
    if merged.len() >= 2
        && merged[0].text.chars().count() == 1
        && !merged[0].text.chars().any(|ch| ch.is_ascii_alphabetic())
        && merged[1].start - merged[0].end <= 0.8
    {
        let first = merged.remove(0);
        merged[0].text.insert_str(0, &first.text);
        merged[0].start = first.start;
    }
    if merged.is_empty() {
        return Err("对齐结果没有可用字幕。".into());
    }
    Ok(merged)
}

fn format_time(value: f64) -> String {
    let total = (value.max(0.0) * 1000.0).round() as u64;
    let hours = total / 3_600_000;
    let minutes = total / 60_000 % 60;
    let seconds = total / 1000 % 60;
    let millis = total % 1000;
    format!("{hours:02}:{minutes:02}:{seconds:02},{millis:03}")
}

pub fn write_srt(segments: &[Segment]) -> Result<String, String> {
    if segments.is_empty() {
        return Err("没有可导出的字幕。".into());
    }
    let mut output = String::new();
    for (index, segment) in segments.iter().enumerate() {
        if !segment.start.is_finite()
            || !segment.end.is_finite()
            || segment.start < 0.0
            || segment.end <= segment.start
            || segment.text.trim().is_empty()
        {
            return Err(format!("第 {} 行字幕的时间或内容无效。", index + 1));
        }
        output.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            index + 1,
            format_time(segment.start),
            format_time(segment.end),
            segment.text.trim()
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_multiline_subtitle() {
        let input = "1\r\n00:00:01,250 --> 00:00:02,500\r\n第一行\r\n第二行\r\n";
        let segments = parse_srt(input).unwrap();
        assert_eq!(segments[0].start, 1.25);
        assert_eq!(segments[0].text, "第一行\n第二行");
        assert_eq!(
            parse_srt(&write_srt(&segments).unwrap()).unwrap()[0].text,
            segments[0].text
        );
    }

    #[test]
    fn splits_aligned_qwen_cues_at_punctuation() {
        let raw = "1\n00:00:00,000 --> 00:00:00,200\n你\n\n2\n00:00:00,200 --> 00:00:00,400\n好\n\n3\n00:00:00,400 --> 00:00:00,400\n。\n\n4\n00:00:01,000 --> 00:00:01,200\n世\n\n5\n00:00:01,200 --> 00:00:01,400\n界\n";
        let lines = split_aligned_srt(raw).unwrap();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "你好");
        assert_eq!(lines[1].text, "世界");
        assert_eq!(lines[1].start, 1.0);
    }

    #[test]
    fn rejects_coarse_qwen_timestamps() {
        let raw = "1\n00:00:00,000 --> 00:00:20,000\n这是一整段没有字级时间轴的很长识别文字，不能伪装成已经对齐的字幕行，必须明确报错。\n";
        assert!(split_aligned_srt(raw).is_err());
    }

    #[test]
    fn whole_transcript_keeps_all_text_in_one_segment() {
        let segments = vec![
            Segment {
                start: 1.0,
                end: 2.0,
                text: "你好，".into(),
            },
            Segment {
                start: 2.0,
                end: 4.0,
                text: "世界。".into(),
            },
            Segment {
                start: 4.0,
                end: 5.0,
                text: "Hello".into(),
            },
            Segment {
                start: 5.0,
                end: 6.0,
                text: "world.".into(),
            },
        ];
        let transcript = whole_transcript(segments).unwrap();
        assert_eq!(transcript.len(), 1);
        assert_eq!(transcript[0].start, 0.0);
        assert_eq!(transcript[0].end, 6.0);
        assert_eq!(transcript[0].text, "你好，世界。Hello world.");
    }

    #[test]
    #[ignore = "requires a local aligned SRT and expected segment count"]
    fn splits_local_aligned_srt() {
        let path = std::env::var("MODEL_SDK_ALIGNED_SRT").unwrap();
        let expected: usize = std::env::var("MODEL_SDK_EXPECTED_SEGMENTS")
            .unwrap()
            .parse()
            .unwrap();
        let content = std::fs::read_to_string(path).unwrap();
        let lines = split_aligned_srt(&content).unwrap();
        assert_eq!(lines.len(), expected);
    }
}
