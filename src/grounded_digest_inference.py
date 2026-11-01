"""Grounded Arabic digest generation using retrieved call transcripts."""

import re


class GroundedDigest:
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self, model_name=None):
        self.model_name = model_name or self.MODEL_NAME
        self.tokenizer = None
        self.model = None

    def load_model(self):
        if self.model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()

    @staticmethod
    def check_citations(digest, sources):
        valid = {f"S{i}" for i in range(1, len(sources) + 1)}
        found = re.findall(r"\[(S\d+)\]", digest)

        return {
            "citations_found": found,
            "invalid_citations": sorted(set(found) - valid),
            "has_at_least_one_citation": bool(found),
            "citation_ids_to_audio": {
                f"S{i}": source["audio_id"]
                for i, source in enumerate(sources, 1)
            },
        }

    def generate(self, query, sources, max_new_tokens=350):
        if not sources:
            raise ValueError("No retrieved sources available")

        self.load_model()

        import torch

        source_block = "\n\n".join(
            f"[S{i}] audio_id={source['audio_id']}\nالنص: {source['text']}"
            for i, source in enumerate(sources, 1)
        )

        system = (
            "أنت مساعد يكتب ملخصًا لمشرف مركز اتصال باللغة العربية. "
            "استخدم المعلومات الموجودة في المصادر فقط. "
            "لا تستنتج سببًا أو نتيجة غير مذكورة. "
            "إذا لم تكفِ المصادر، قل إن المعلومات غير كافية. "
            "اكتب 3 إلى 5 نقاط موجزة، وكل نقطة يجب أن تنتهي باستشهاد مثل [S1]. "
            "لا تخترع أرقام تسجيلات أو استشهادات."
        )

        user = (
            f"موضوع الملخص: {query}\n\n"
            f"المصادر المتاحة:\n{source_block}\n\n"
            "اكتب الملخص."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1]:]

        digest = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        return {
            "digest_text": digest,
            "sources": sources,
            "citation_check": self.check_citations(digest, sources),
        }
