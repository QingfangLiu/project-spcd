"""Regression checks for content preservation and failure without overwriting."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from convert_pnas import ConversionError, convert, group, main, strip_comments
from build_simple import clear_incompatible_bibliography_cache


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

SUPPLEMENT = r"""\documentclass[9pt,twoside,lineno]{pnas-new}
\templatetype{pnassupportinginfo}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{placeins}
\title{A supplementary manuscript}
\author{Author One and Author Two}
\correspondingauthor{Author One.\\E-mail: one@example.org}
\renewcommand{\thefigure}{S\arabic{figure}}
\begin{document}
\maketitle
\SItext
\subsection*{Supplementary Methods}
Methods must survive unchanged, with \cite{key1} and Fig.~\ref{fig:supp}.
\subsection*{Supplementary Results}
Results with $p < 0.05$.
\clearpage
\begin{figure}[p]
\centering
\includegraphics[width=\linewidth]{fig/one.pdf}
\caption{A supplementary caption with \textbf{formatting}.}
\label{fig:supp}
\end{figure}
\FloatBarrier
\bibliography{references}
\end{document}
"""


class ConversionTests(unittest.TestCase):
    def test_supplement_without_abstract_preserves_content_and_numbering(self):
        output, report = convert(SUPPLEMENT, TEMPLATE, "supp")
        for text in [r"\subsection*{Supplementary Methods}",
                     r"\subsection*{Supplementary Results}",
                     r"Methods must survive unchanged, with \citep{key1} and Fig.~\ref{fig:supp}.",
                     r"Results with $p < 0.05$.",
                     r"A supplementary caption with \textbf{formatting}.",
                     r"\renewcommand{\thefigure}{S\arabic{figure}}",
                     r"\renewcommand{\thetable}{S\arabic{table}}",
                     r"\bibliographystyle{plainnat}", r"\clearpage"]:
            self.assertIn(text, output)
        for text in [r"\section*{Introduction}", r"\section*{Abstract}",
                     r"\SItext", r"\FloatBarrier", r"\begin{figure}", "pnas-new"]:
            self.assertNotIn(text, output)
        self.assertEqual(report["document_kind"], "supp")
        self.assertTrue(all(x["preserved"] for x in report["checks"].values()))

    def test_supplement_rejects_custom_setup_and_external_content(self):
        for old, replacement in [
            (r"\usepackage{amsmath}", r"\usepackage{unreviewed}"),
            (r"\renewcommand{\thefigure}{S\arabic{figure}}", r"\renewcommand{\thefigure}{X}"),
            (r"\SItext", r"\input{extra}"),
            (r"\templatetype{pnassupportinginfo}", r"\templatetype{pnasresearcharticle}"),
        ]:
            with self.subTest(replacement=replacement), self.assertRaises(ConversionError):
                convert(SUPPLEMENT.replace(old, replacement), TEMPLATE, "supp")

    def test_supplement_png_export_links_assets_without_changing_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir = root / "writing" / "supp"
            source_dir.mkdir(parents=True)
            (root / "supp_fig").mkdir()
            pdf = root / "supp_fig" / "one.pdf"
            pdf.write_bytes(b"Original PDF artwork")
            png = pdf.with_suffix(".png")
            png.write_bytes(b"PNG artwork")
            (source_dir / "references.bib").write_text("@article{key1, title={Reference}}")
            source = source_dir / "supp.tex"
            source_text = SUPPLEMENT.replace("fig/one.pdf", "../../supp_fig/one.pdf")
            source.write_text(source_text)
            output = root / "writing" / "word_export" / "supp.tex"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(source), "-o", str(output), "--document-kind", "supp",
                                       "--asset-mode", "link", "--figure-format", "png"]), 0)
            self.assertIn("{../../supp_fig/one.png}", output.read_text())
            self.assertIn(r"\bibliography{../supp/references}", output.read_text())
            self.assertEqual(source.read_text(), source_text)
            self.assertEqual(pdf.read_bytes(), b"Original PDF artwork")
            self.assertFalse((output.parent / "supp_fig").exists())

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
                     r"Author One\textsuperscript{a,1}", r"\citep{key3}"]:
            self.assertIn(text, output)
        self.assertIn(r"\section*{Materials and methods}", output)
        self.assertNotIn(r"\begin{figure*}", output)
        self.assertNotIn(r"\matmethods", output)
        self.assertIn(r"\refstepcounter{figure}", output)
        self.assertIn(r"\refstepcounter{table}", output)
        self.assertEqual(report["checks"]["citations"]["count"], 2)
        self.assertEqual(report["checks"]["captions"]["count"], 2)
        self.assertTrue(all(x["preserved"] for x in report["checks"].values()))

    def test_author_year_citations_preserve_multiple_keys_and_optional_notes(self):
        source = SOURCE.replace(r"\cite{key1,key2}", r"\cite[see][pp.~{2--3}]{key1,key2}")
        output, report = convert(source, TEMPLATE)
        self.assertIn(r"\citep[see][pp.~{2--3}]{key1,key2}", output)
        self.assertIn(r"\usepackage[authoryear,round,sort]{natbib}", output)
        self.assertIn(r"\bibliographystyle{plainnat}", output)
        self.assertEqual(report["citation_style"], "author-year")
        self.assertEqual(report["checks"]["citations"]["count"], 2)
        self.assertEqual(report["citation_rewrites"], {"parenthetical": 2, "narrative_year": 0})

    def test_existing_narrative_authors_in_body_and_caption_get_year_only(self):
        source = SOURCE.replace(r"\cite{key1,key2}", r"Bartra et al.\ \cite{key1}")
        source = source.replace("A caption with", r"Bartra et al.~\cite{key2}. A caption with")
        output, report = convert(source, TEMPLATE)
        self.assertIn(r"Bartra et al.\ \citeyearpar{key1}", output)
        self.assertIn(r"Bartra et al.~\citeyearpar{key2}. A caption with", output)
        self.assertEqual(report["citation_rewrites"], {"parenthetical": 1, "narrative_year": 2})
        self.assertTrue(report["checks"]["captions"]["preserved"])

    def test_highlights_preserve_abstract_heading_math_and_citations(self):
        source = SOURCE.replace(r"\begin{document}",
                                "\\usepackage{soul}\n\\usepackage{xcolor}\n\\begin{document}")
        source = source.replace("An abstract with", r"An abstract \hl{with highlighted wording} with")
        source = source.replace(r"\section*{Discussion}", r"\section*{\hl{Discussion}}")
        source = source.replace("Original conclusion.",
                                r"\hl{An error cost \mbox{($D_i$)} and \mbox{\citealp{key1,key2}}.}")
        source = source.replace("Methods sentence that must never be lost, with \\cite{key3}.",
                                r"\hl{Methods with \mbox{\cite{key3}} must never be lost.}")
        output, report = convert(source, TEMPLATE)
        for text in [r"\hl{with highlighted wording}", r"\section*{\hl{Discussion}}",
                     r"\hl{An error cost \mbox{($D_i$)} and \mbox{\citealp{key1}}; \mbox{\citealp{key2}}.}",
                     r"\hl{Methods with \mbox{\citep{key3}} must never be lost.}",
                     r"\DeclareRobustCommand{\hl}{\textcolor{reviewtext}}",
                     r"\usepackage{xcolor}"]:
            self.assertIn(text, output)
        self.assertEqual(report["checks"]["highlights"], {"count": 4, "preserved": True})
        self.assertEqual(report["checks"]["citations"]["count"], 3)
        self.assertEqual(report["highlighted_citation_reflows"], 1)

    def test_highlighted_citation_lists_wrap_without_changing_notes_or_other_citations(self):
        source = SOURCE.replace("Original conclusion.",
                                r"\hl{Compare \mbox{\cite{key1,key2,key3,key4}}.} "
                                r"\hl{With notes \mbox{\cite[see][p.~2]{key1,key2}}.} "
                                r"Unhighlighted \mbox{\cite{key1,key2}}.")
        output, report = convert(source, TEMPLATE)
        self.assertIn(r"\hl{Compare (\mbox{\citealp{key1}}; \mbox{\citealp{key2}}; "
                      r"\mbox{\citealp{key3}}; \mbox{\citealp{key4}}).}", output)
        self.assertIn(r"\hl{With notes \mbox{\citep[see][p.~2]{key1,key2}}.}", output)
        self.assertIn(r"Unhighlighted \mbox{\citep{key1,key2}}.", output)
        self.assertEqual(report["highlighted_citation_reflows"], 1)

    def test_highlight_packages_are_supported_for_supplements(self):
        source = SUPPLEMENT.replace(r"\usepackage{amsmath}",
                                    "\\usepackage{amsmath}\n\\usepackage{soul,xcolor}")
        source = source.replace("Results with $p < 0.05$.", r"\hl{Results with \mbox{$p < 0.05$}.}")
        output, report = convert(source, TEMPLATE, "supp")
        self.assertIn(r"\hl{Results with \mbox{$p < 0.05$}.}", output)
        self.assertEqual(report["checks"]["highlights"], {"count": 1, "preserved": True})

    def test_highlight_support_still_rejects_unreviewed_preamble(self):
        for kind, source in [("main", SOURCE), ("supp", SUPPLEMENT)]:
            for insertion in [r"\usepackage[unreviewed]{soul}",
                              r"\usepackage{soul,unreviewed}", r"\sethlcolor{red}"]:
                with self.subTest(kind=kind, insertion=insertion), self.assertRaises(ConversionError):
                    convert(source.replace(r"\begin{document}", insertion + "\n"
                                           + r"\begin{document}"), TEMPLATE, kind)

    def test_unparenthesized_citation_keys_and_notes_are_checked(self):
        source = SOURCE.replace(r"\cite{key1,key2}", r"\citealp[see][pp.~{2--3}]{key1,key2}")
        output, report = convert(source, TEMPLATE)
        self.assertIn(r"\citealp[see][pp.~{2--3}]{key1,key2}", output)
        self.assertEqual(report["checks"]["citations"], {"count": 2, "preserved": True})

    def test_bibliography_cache_migration_keeps_pdf_and_matching_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "main.tex"
            for suffix in (".pdf", ".bbl", ".blg", ".fdb_latexmk"):
                output.with_suffix(suffix).write_bytes(b"Keep final output")
            output.with_suffix(".aux").write_text(r"\bibstyle{unsrt}")
            clear_incompatible_bibliography_cache(output, "plainnat")
            self.assertEqual(output.with_suffix(".pdf").read_bytes(), b"Keep final output")
            self.assertFalse(output.with_suffix(".bbl").exists())
            output.with_suffix(".aux").write_text(r"\bibstyle{plainnat}")
            output.with_suffix(".bbl").write_bytes(b"Compatible bibliography")
            clear_incompatible_bibliography_cache(output, "plainnat")
            self.assertEqual(output.with_suffix(".bbl").read_bytes(), b"Compatible bibliography")

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
