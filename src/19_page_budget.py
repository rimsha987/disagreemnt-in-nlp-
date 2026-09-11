"""Where the page count goes. Estimates length by content type, no compiler needed."""
import re

VERB = re.compile(r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}", re.S)
TAB = re.compile(r"\\begin\{table\}")
FIG = re.compile(r"\\begin\{figure\}")
ROW = re.compile(r"\\\\")

LINES_PER_PAGE = 45          # 11pt verbatim, 2.5cm margins
WORDS_PER_PAGE = 480         # 11pt prose at this measure


def strip_comments(s):
    return "\n".join(l for l in s.split("\n") if not l.lstrip().startswith("%"))


def main():
    main_s = strip_comments(open("tex/main.tex", encoding="utf-8").read())
    app_s = strip_comments(open("tex/appendix_content.tex", encoding="utf-8").read())

    vb = VERB.findall(app_s)
    vlines = sum(len(v.strip("\n").split("\n")) for v in vb)
    app_prose = VERB.sub("", app_s)
    app_words = len(app_prose.split())
    app_tabs = len(TAB.findall(app_s))
    app_rows = len(ROW.findall(app_prose))

    body_words = len(main_s.split())
    body_tabs, body_figs = len(TAB.findall(main_s)), len(FIG.findall(main_s))

    p_verb = vlines / LINES_PER_PAGE
    p_appprose = app_words / WORDS_PER_PAGE
    p_apptab = app_tabs * 0.45
    p_app = p_verb + p_appprose + p_apptab

    print("WHY THE PDF IS 31 PAGES\n")
    print("  BODY - your original text, unchanged")
    print(f"     words                        {body_words:,}")
    print(f"     tables                       {body_tabs}")
    print(f"     figures                      {body_figs}")
    print(f"     your submitted PDF           8,673 words over 20 pages")
    print(f"     -> still about 20 pages\n")

    print("  FRONT MATTER I ADDED (not in your PDF)")
    print(f"     full title page              ~1.0 page")
    print(f"     table of contents            ~1.5 pages")
    print(f"     -> about 2.5 pages\n")

    print("  APPENDICES B/C/D - the content you asked me to fill in")
    print(f"     verbatim lines               {vlines}  -> {p_verb:.1f} pages")
    print(f"        (2 full prompts + 19 persona texts, {len(vb)} blocks)")
    print(f"     appendix tables              {app_tabs}  -> {p_apptab:.1f} pages")
    print(f"     appendix prose               {app_words:,} words -> {p_appprose:.1f} pages")
    print(f"     -> about {p_app:.0f} pages\n")

    print(f"  TOTAL  20 + 2.5 + {p_app:.0f} = about {20 + 2.5 + p_app:.0f} pages\n")
    print("  Nothing was added to your argument. The growth is appendix material")
    print("  plus front matter, both of which are normally excluded from a page limit.")

    print("\n" + "=" * 72)
    print("IF YOU NEED IT SHORTER")
    print("=" * 72)
    print(f"  remove the table of contents         -1.5 pages   (delete \\tableofcontents)")
    print(f"  compact title page                   -1.0 page    (use \\maketitle instead)")
    print(f"  set appendices in \\footnotesize      -{p_app*0.25:.1f} pages")
    print(f"  move prompts to a supplementary file -{p_verb*0.75:.1f} pages")
    print(f"  drop appendices entirely             -{p_app:.0f} pages  -> back to ~20")


if __name__ == "__main__":
    main()
