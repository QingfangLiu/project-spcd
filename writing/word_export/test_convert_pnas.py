"""Regression checks for content preservation and failure without overwriting."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from convert_pnas import ConversionError, convert, group, main, strip_comments


TEMPLATE = Path(__file__).with_name("simple_template.tex").read_text()
SOURCE = r"""\documentclass[9pt,twocolumn]{pnas-new}
\articletype{NEUROSCIENCE}
\begin{document}
\title{A title with \textbf{nested braces}}
\author[a,1]{Author One}
\author[b]{Author Two}
\affil[a]{Institute \& Center}
\affil[b]{Other institute}
\significancestatement{A meaningful statement with \emph{emphasis}.}
\authorcontributions{A.O. designed research.}
\authordeclaration{No competing interests.}
\correspondingauthor{\textsuperscript{1}one@example.org}
\keywords{alpha $|$ beta $|$ gamma}
\begin{abstract}An abstract with a 95\% interval.\end{abstract}
\dates{Compiled on \today}
\doi{\url{www.pnas.org/placeholder}}
\maketitle
\ifthenelse{\boolean{shortarticle}}{\ifthenelse{\boolean{singlecolumn}}{\abscontentformatted}{\abscontent}}{}
\Firstpage
Introduction sentence with \cite{key1,key2}. Figure~\ref{fig:one}.
\Parasplit
Another sentence with an escaped brace: \{example\}.
\begin{figure*}[t]
\centering
\includegraphics[width=0.95\linewidth]{fig/one.pdf}
\caption{A caption with \textbf{nested {formatting}} and 95\%.}
\label{fig:one}
\end{figure*}
\begin{equation}L_i = a_i e^{-r_i}.\end{equation}
\section*{Discussion}
Original conclusion.
\matmethods{
\subsection*{Participants}
Methods sentence that must never be lost, with \cite{key3}.
\begin{table}[t]
\caption{A table caption.}\label{tab:one}
\centering
\begin{tabular}{cc}Name & Value \\ Alpha & $a_i$ \\\end{tabular}
\end{table}
}
\showmatmethods{}
\dataavail{Data are available on request.}
\acknow{We thank \textbf{our colleagues}.}
\showacknow{}
\bibsplit[17]
\bibliography{references}
\end{document}
"""


class ConversionTests(unittest.TestCase):
    def test_nested_groups_and_optional_arguments(self):
        value, end = group(r" {a {b} \{c\}} trailing", 0)
        self.assertEqual(value, r"a {b} \{c\}")
        self.assertEqual(r" {a {b} \{c\}} trailing"[end:], " trailing")
        self.assertEqual(group("[x={]}]", 0, "[")[0], "x={]}")

    def test_comments_do_not_consume_escaped_percent(self):
        self.assertEqual(strip_comments("A\\% B% comment\nC"), "A\\% BC")
        self.assertEqual(strip_comments("A\\\\% comment\nC"), "A\\\\C")

    def test_relocates_all_content_and_preserves_objects(self):
        output, report = convert(SOURCE, TEMPLATE)
        for text in ["Methods sentence that must never be lost", "Original conclusion.",
                     r"A caption with \textbf{nested {formatting}} and 95\%.",
                     r"We thank \textbf{our colleagues}.",
                     r"Author One\textsuperscript{a,1}", r"\cite{key3}"]:
            self.assertIn(text, output)
        self.assertIn(r"\section*{Materials and methods}", output)
        self.assertNotIn(r"\begin{figure*}", output)
        self.assertNotIn(r"\matmethods", output)
        self.assertIn(r"\refstepcounter{figure}", output)
        self.assertIn(r"\refstepcounter{table}", output)
        self.assertEqual(report["checks"]["citations"]["count"], 2)
        self.assertEqual(report["checks"]["captions"]["count"], 2)
        self.assertTrue(all(x["preserved"] for x in report["checks"].values()))

    def test_front_matter_can_also_be_in_preamble(self):
        source = SOURCE.replace(r"\title{A title with \textbf{nested braces}}", "")
        source = source.replace(r"\begin{document}",
                                r"\title{A title with \textbf{nested braces}}" + "\n"
                                + r"\begin{document}")
        output, _ = convert(source, TEMPLATE)
        self.assertIn(r"A title with \textbf{nested braces}", output)

    def test_unsupported_input_fails_instead_of_dropping_it(self):
        for insertion in [r"\input{methods}", r"\newcommand{\custom}{something}"]:
            with self.subTest(insertion=insertion), self.assertRaises(ConversionError):
                convert(SOURCE.replace(r"\begin{document}", insertion + "\n"
                                       + r"\begin{document}"), TEMPLATE)

    def test_unknown_conditional_is_not_silently_deleted(self):
        with self.assertRaises(ConversionError):
            convert(SOURCE.replace(r"\Firstpage", r"\ifthenelse{true}{Important text}{Other text}"), TEMPLATE)

    def test_external_input_inside_methods_is_also_rejected(self):
        with self.assertRaises(ConversionError):
            convert(SOURCE.replace("Methods sentence", r"\input{extra} Methods sentence"), TEMPLATE)

    def test_missing_assets_and_repeat_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.tex"
            output = root / "converted" / "main.tex"
            source.write_text(SOURCE)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(source), "-o", str(output)]), 0)
            original = output.read_bytes()
            self.assertIn('"status": "missing"', output.with_suffix(".conversion.json").read_text())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(source), "-o", str(output)]), 1)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(source.read_text(), SOURCE)

    def test_conflicting_assets_stop_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.tex"
            source.write_text(SOURCE)
            for folder, content in [("first", b"first PDF"), ("second", b"different PDF")]:
                (root / folder).mkdir()
                (root / folder / "one.pdf").write_bytes(content)
            output = root / "converted" / "main.tex"
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(source), "-o", str(output),
                                       "--asset-dir", str(root / "first"),
                                       "--asset-dir", str(root / "second")]), 1)
            self.assertFalse(output.exists())

    def test_shared_assets_and_source_refresh_without_copies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir = root / "writing" / "main"
            source_dir.mkdir(parents=True)
            (root / "fig").mkdir()
            figure = root / "fig" / "one.pdf"
            figure.write_bytes(b"shared figure bytes")
            bibliography = source_dir / "references.bib"
            bibliography.write_text("@article{key1, title={Shared reference}}")
            source = source_dir / "main.tex"
            source_text = SOURCE.replace("fig/one.pdf", "../../fig/one.pdf")
            source.write_text(source_text)
            output = root / "writing" / "simple" / "main.tex"
            arguments = [str(source), "-o", str(output), "--asset-mode", "link", "--overwrite"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            self.assertIn(r"{../../fig/one.pdf}", output.read_text())
            self.assertIn(r"\bibliography{../main/references}", output.read_text())
            self.assertFalse((output.parent / "fig").exists())
            self.assertFalse((output.parent / "references.bib").exists())
            report = json.loads(output.with_suffix(".conversion.json").read_text())
            self.assertTrue(report["compilation_ready"])
            self.assertTrue(all(item["status"] == "linked" for item in report["assets"]))
            source.write_text(source_text.replace("Original conclusion.", "Revised authoritative conclusion."))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            self.assertIn("Revised authoritative conclusion.", output.read_text())
            self.assertNotIn("Original conclusion.", output.read_text())
            self.assertEqual(figure.read_bytes(), b"shared figure bytes")
            self.assertEqual(bibliography.read_text(), "@article{key1, title={Shared reference}}")

    def test_missing_shared_asset_does_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.tex"
            source.write_text(SOURCE)
            output = root / "main.tex"
            output.write_text("Preserve previous output")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(source), "-o", str(output), "--asset-mode", "link", "--overwrite"]), 1)
            self.assertEqual(output.read_text(), "Preserve previous output")
            self.assertFalse(output.with_suffix(".conversion.json").exists())

    def test_png_figures_in_link_and_copy_modes(self):
        for mode in ("link", "copy"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "fig").mkdir()
                (root / "fig" / "one.pdf").write_bytes(b"PDF artwork")
                png = root / "fig" / "one.png"
                png.write_bytes(b"PNG artwork")
                (root / "references.bib").write_text("@article{key1, title={Reference}}")
                source = root / "source.tex"
                source.write_text(SOURCE)
                output = root / "converted" / "main.tex"
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main([str(source), "-o", str(output),
                                           "--asset-mode", mode, "--figure-format", "png"]), 0)
                expected = "../fig/one.png" if mode == "link" else "fig/one.png"
                self.assertIn("{" + expected + "}", output.read_text())
                self.assertNotIn("{fig/one.pdf}", output.read_text())
                report = json.loads(output.with_suffix(".conversion.json").read_text())
                self.assertEqual(report["figure_format"], "png")
                self.assertEqual(report["assets"][0]["origin"], str(png.resolve()))
                self.assertEqual(report["assets"][0]["source_path"], "fig/one.pdf")
                self.assertTrue(all(check["preserved"] for check in report["checks"].values()))
                if mode == "copy":
                    self.assertEqual((output.parent / expected).read_bytes(), png.read_bytes())
                    self.assertFalse((output.parent / "fig" / "one.pdf").exists())
                else:
                    self.assertFalse((output.parent / "fig").exists())
                self.assertEqual(source.read_text(), SOURCE)

    def test_missing_png_does_not_overwrite_previous_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "fig").mkdir()
            (root / "fig" / "one.pdf").write_bytes(b"PDF artwork")
            source = root / "source.tex"
            source.write_text(SOURCE)
            output = root / "converted" / "main.tex"
            output.parent.mkdir()
            output.write_text("Previous output")
            report = output.with_suffix(".conversion.json")
            report.write_text("Previous report")
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(main([str(source), "-o", str(output), "--asset-mode", "link",
                                       "--figure-format", "png", "--overwrite"]), 1)
            self.assertIn("Matching PNG figure not found", errors.getvalue())
            self.assertEqual(output.read_text(), "Previous output")
            self.assertEqual(report.read_text(), "Previous report")

    def test_copy_mode_still_rejects_parent_relative_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.tex"
            source.write_text(SOURCE.replace("fig/one.pdf", "../fig/one.pdf"))
            output = root / "converted" / "main.tex"
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(source), "-o", str(output)]), 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
