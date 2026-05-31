# How To Scale Your Model

This book aims to demystify the art of scaling LLMs on TPUs. We try to explain how TPUs work, how LLMs actually run at scale, and how to pick parallelism schemes during training and inference that avoid communication bottlenecks. The book is available at https://jax-ml.github.io/scaling-book.

### Acknowledgments

This book was written by Jacob Austin, Sholto Douglas, Roy Frostig, Anselm Levskaya, Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy, Vinay Ramasesh and Albert Webson at Google DeepMind. Many of the ideas were first derived by James Bradbury and Reiner Pope.

The website uses a Distill-style Jekyll theme created by https://github.com/alshedivat/al-folio and the Distill team. Thank you!

### Running Locally

To build this repo locally, you will need Ruby, ImageMagick, and Jupyter installed, which for MacOS can be installed with Homebrew using

```
brew install imagemagick ruby
pip install jupyter
```

After this is installed, you should make sure the correct version of Ruby is found in PATH. You should have at least Ruby 3.4.5 installed. Then you ought to be able to run

```
git clone https://github.com/jax-ml/scaling-book.git
cd scaling-book
bundle install
bundle exec jekyll serve
```

Once you have run jekyll serve successfully, the book will be available at `http://127.0.0.1:4000/scaling-book`.

The Github Pages deployment is handled by a GitHub Action that runs automatically on new commits to the main branch.

### Adding Translations

Translated pages live under a language-specific directory while the original English pages keep their existing URLs. For example, a Korean translation of `training.md` should be added as:

```text
training.md
ko/training.md
```

Add a shared `translation_key` to the original English page frontmatter:

```yaml
lang: en
translation_key: training
```

The translated page should use the same `translation_key`, set its language, and include translation metadata:

```yaml
---
layout: distill
lang: ko
translation_key: training
permalink: /ko/training/
title: "훈련을 위해 Transformer를 병렬화하는 법"
description: "..."
date: 2025-02-04

translation:
  date: 2026-05-29
  translators:
    - name: Translator Name
      url: https://example.com
  reviewers:
    - name: Reviewer Name
      url: https://example.com
---
```

If a translation has not been reviewed yet, omit `reviewers`; the page will show it as not yet reviewed. Keep the original Markdown structure as much as possible, including figure includes, `<d-footnote>`, `<d-cite>`, math, and bibliography references.

For links to other pages in this book, point to the translated page when it exists. If it does not exist yet, point back to the original English page with a relative parent link, such as `../roofline` or `../conclusion#further-reading`.

Before sending a translation PR, run:

```bash
bundle exec jekyll build
```

Then check the translated page in a browser. Confirm that the language dropdown links between the original and translation, the translator/reviewer/date block appears under the authors, and figures, citations, footnotes, math, and internal links still work.

### Generating a Single Document

To combine all chapters into a single markdown file with standardized formatting:

```bash
python bin/convert_to_single_md.py
```

This generates `scaling-book-combined.md` in the repository root. The script:
- Strips Jekyll frontmatter
- Converts `{% include figure.liquid %}` to standard markdown images
- Converts internal page links to anchor links
- Converts inline `$$` math to `$`
- Strips unsupported LaTeX commands (with warnings)

To convert the combined markdown to a Word document:

```bash
pandoc scaling-book-combined.md -o scaling-book.docx
```

### Contributing and Contact

If you see any issues or have questions, please leave a comment on the website itself (powered by Giscus) or in the GitHub discussion. Feel free to send a PR if you want to contribute. You can also email jaaustin [at] google [dot] com.

To contribute on GitHub you will need to sign a Google "Contributor License Agreement" (CLA). You can do that here: https://cla.developers.google.com/clas.

### Citation

For attribution in academic contexts, please cite this work as

```Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.```

BibTeX citation

```
@article{scaling-book,
  title = {How to Scale Your Model},
  author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya, Anselm and Chen, Charlie and Vikram, Sharad and Lebron, Federico and Choy, Peter and Ramasesh, Vinay and Webson, Albert and Pope, Reiner},
  publisher = {Google DeepMind},
  howpublished = {Online},
  note = {Retrieved from https://jax-ml.github.io/scaling-book/},
  year = {2025}
}
```

![dragon](assets/img/dragon.png)

*This book was originally called "How To Scale Your Dragon", after the Dreamworks film, hence the dragon imagery.*
