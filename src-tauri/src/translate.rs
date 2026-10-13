use model_sdk::{
    subtitle::{write_srt, Segment},
    task::TaskControl,
    ChatMessage, ChatRole, GenerationOptions, ModelSdk, LLAMA_ENGINE_ID,
};

fn language_name(code: &str) -> Option<&'static str> {
    match code {
        "zh" => Some("Simplified Chinese"),
        "en" => Some("English"),
        "ja" => Some("Japanese"),
        "ko" => Some("Korean"),
        "yue" => Some("Cantonese"),
        "fr" => Some("French"),
        "de" => Some("German"),
        "es" => Some("Spanish"),
        "pt" => Some("Portuguese"),
        "ru" => Some("Russian"),
        "ar" => Some("Arabic"),
        "th" => Some("Thai"),
        "vi" => Some("Vietnamese"),
        "id" => Some("Indonesian"),
        "ms" => Some("Malay"),
        _ => None,
    }
}

pub(crate) fn translate_segments(
    sdk: &ModelSdk,
    model_id: &str,
    target_language: &str,
    segments: &[Segment],
    task: &TaskControl,
) -> Result<Vec<Segment>, String> {
    let target = language_name(target_language).ok_or("不支持的目标语言。")?;
    write_srt(segments)?;
    let options = GenerationOptions {
        max_tokens: 256,
        temperature: Some(0.0),
        top_k: Some(20),
        top_p: Some(0.6),
        repeat_penalty: Some(1.05),
        ..GenerationOptions::default()
    };
    let mut translated = Vec::with_capacity(segments.len());
    for (index, segment) in segments.iter().enumerate() {
        task.check_cancelled()?;
        task.progress(
            "translate",
            (index * 100 / segments.len()) as u8,
            &format!("正在翻译第 {} / {} 条", index + 1, segments.len()),
        );
        let previous = segments[index.saturating_sub(2)..index]
            .iter()
            .map(|item| item.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let prompt = translation_prompt(&segment.text, target, &previous, target_language == "zh");
        let mut text = sdk.generate_chat(
            LLAMA_ENGINE_ID,
            model_id,
            &[ChatMessage {
                role: ChatRole::User,
                content: prompt,
            }],
            &options,
            task,
        )?;
        if !looks_like_target(&text, target_language) {
            let retry = format!("Translate this text into {target}. Output {target} only; do not repeat or paraphrase the source text.\nText: {}", segment.text);
            text = sdk.generate_chat(
                LLAMA_ENGINE_ID,
                model_id,
                &[ChatMessage {
                    role: ChatRole::User,
                    content: retry,
                }],
                &options,
                task,
            )?;
        }
        if !looks_like_target(&text, target_language) {
            return Err(format!("第 {} 条字幕未得到目标语言的译文。", index + 1));
        }
        translated.push(Segment {
            start: segment.start,
            end: segment.end,
            text: text.trim().into(),
        });
    }
    task.progress("translate", 100, "字幕翻译完成");
    Ok(translated)
}

fn looks_like_target(text: &str, target: &str) -> bool {
    let contains = |start, end| text.chars().any(|ch| (start..=end).contains(&ch));
    match target {
        "zh" | "yue" => contains('\u{4e00}', '\u{9fff}'),
        "ja" => contains('\u{3040}', '\u{30ff}') || contains('\u{4e00}', '\u{9fff}'),
        "ko" => contains('\u{ac00}', '\u{d7af}'),
        "ru" => contains('\u{0400}', '\u{04ff}'),
        "ar" => contains('\u{0600}', '\u{06ff}'),
        "th" => contains('\u{0e00}', '\u{0e7f}'),
        _ => text
            .chars()
            .any(|ch| ch.is_ascii_alphabetic() || ('\u{00c0}'..='\u{024f}').contains(&ch)),
    }
}

fn translation_prompt(text: &str, target: &str, previous: &str, chinese: bool) -> String {
    let chinese = chinese
        || text
            .chars()
            .any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch));
    if chinese {
        if previous.is_empty() {
            format!(
                "将以下文本翻译为{target}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
            )
        } else {
            format!("{previous}\n参考上面的信息，把下面的文本翻译成{target}，注意不需要翻译上文，也不要额外解释：\n{text}")
        }
    } else {
        if previous.is_empty() {
            format!("Translate the following segment into {target}, without additional explanation.\n\n{text}")
        } else {
            format!("{previous}\nRefer to the information above and translate the following text into {target}. Do not translate the context above and do not add explanations:\n{text}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{language_name, looks_like_target, translation_prompt};

    #[test]
    fn translation_languages_match_the_desktop_choices() {
        for code in [
            "zh", "en", "ja", "ko", "yue", "fr", "de", "es", "pt", "ru", "ar", "th", "vi", "id",
            "ms",
        ] {
            assert!(language_name(code).is_some());
        }
        assert!(language_name("unknown").is_none());
    }

    #[test]
    fn prompt_marks_previous_subtitles_as_context_only() {
        let prompt = translation_prompt("你好", "English", "早上好", false);
        assert!(prompt.contains("不需要翻译上文"));
        assert!(prompt.ends_with("你好"));
        let prompt = translation_prompt("Hello", "French", "Good morning", false);
        assert!(prompt.contains("Do not translate the context above"));
        assert!(prompt.ends_with("Hello"));
    }

    #[test]
    fn rejects_a_source_language_paraphrase() {
        assert!(!looks_like_target("今天的天气非常好。", "en"));
        assert!(looks_like_target("The weather is very nice today.", "en"));
        assert!(looks_like_target("今天的天气非常好。", "zh"));
        assert!(!looks_like_target("The weather is very nice today.", "zh"));
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "requires MODEL_SDK_LLAMA_ENGINE and MODEL_SDK_LLAMA_MODEL"]
    fn translates_with_local_model_and_keeps_timestamps() {
        use super::*;
        use std::{fs, path::Path};

        let engine = std::env::var("MODEL_SDK_LLAMA_ENGINE").unwrap();
        let model = std::env::var("MODEL_SDK_LLAMA_MODEL").unwrap();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("subtitle-translation-smoke-{nonce}"));
        let sdk = ModelSdk::new(&root).unwrap();
        sdk.register_local_engine(LLAMA_ENGINE_ID, Path::new(&engine))
            .unwrap();
        sdk.register_local_model(
            model_sdk::ModelDescriptor {
                id: "translation-test".into(),
                label: "Translation test".into(),
                format: "llama-gguf".into(),
                capabilities: vec![model_sdk::TEXT_GENERATION.into()],
            },
            Path::new(&model),
        )
        .unwrap();
        let source = vec![
            Segment {
                start: 0.0,
                end: 2.0,
                text: "今天天气很好。".into(),
            },
            Segment {
                start: 2.0,
                end: 4.0,
                text: "我们出去散步吧。".into(),
            },
        ];
        let translated = translate_segments(
            &sdk,
            "translation-test",
            "en",
            &source,
            &TaskControl::default(),
        )
        .unwrap();
        assert_eq!(translated.len(), source.len());
        for (original, result) in source.iter().zip(translated.iter()) {
            println!("{} -> {}", original.text, result.text);
            assert_eq!((result.start, result.end), (original.start, original.end));
            assert!(!result.text.trim().is_empty());
            assert_ne!(result.text, original.text);
            assert!(result.text.chars().any(|ch| ch.is_ascii_alphabetic()));
            assert!(!result
                .text
                .chars()
                .any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch)));
        }
        drop(sdk);
        fs::remove_dir_all(root).unwrap();
    }
}
