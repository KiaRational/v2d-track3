%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% nvidiaextabstract.cls
%
% Two-page extended abstract for the From Human Videos to Dexterous Robot
% Capability workshop (CoRL 2026): title + "Extended Abstract" subtitle,
% authors, abstract, two-column body, acknowledgments, references and an
% optional appendix, using the look of the NVIDIA tech report template.
% The two-page limit excludes references, acknowledgments and appendices.
%
% This is a thin wrapper around nvidiatechreport.cls, which is used
% unchanged. Keep both files next to your main .tex file.
%
% Usage:
%   \documentclass[10pt, logo, copyright]{nvidiaextabstract}             % camera-ready
%   \documentclass[10pt, logo, copyright, anonymous]{nvidiaextabstract}  % blind review
%
% Options:
%   - anonymous: replaces the author block with the submission ID, hides
%                the NVIDIA logo, the copyright line and the acknowledgments.
%   - any other option is passed on to nvidiatechreport (10pt, logo,
%     copyright, address, internal, ...). Two-column mode is always on.
%
% Extra preamble commands (all optional unless noted):
%   \subtitle{...}       subtitle under the title (default: Extended Abstract)
%   \shorttitle{...}     running head on page 2 (default: the title)
%   \venue{...}          venue line in the first-page footer
%   \submissionID{...}   printed instead of the authors in anonymous mode
%   \logo[opts]{file}    first-page logo; opts go to \includegraphics
%                        (default: \logo[width=90pt]{assets/nvlogo2.png})
%
% Extra document commands/environments:
%   \begin{acks}...\end{acks}   acknowledgments (hidden when anonymous)
%   \Description[short]{long}   accessibility description for a figure;
%                               accepted and not printed
%   \balance                    balances the columns on the last page
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{nvidiaextabstract}

%%======== Options and base class ========
\newif\if@anonymous
\@anonymousfalse
\DeclareOption{anonymous}{\@anonymoustrue}
\DeclareOption{twocolumn}{}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{nvidiatechreport}}
\ProcessOptions\relax
\PassOptionsToClass{twocolumn}{nvidiatechreport}
\LoadClass{nvidiatechreport}

% The first-page footer is driven by the "copyright" flag of the base class.
% Keep it on so the venue line can always be shown, and remember whether
% the NVIDIA copyright line itself was requested.
\newif\if@nvcopyright
\ifthenelse{\boolean{copyright}}{\@nvcopyrighttrue}{\@nvcopyrightfalse}
\if@anonymous
  \@nvcopyrightfalse
  \setboolean{logo}{false}
  \setboolean{address}{false}
\fi
\setboolean{copyright}{true}

%%======== Packages ========
\RequirePackage[numbers,sort&compress]{natbib}
\RequirePackage{balance}
\RequirePackage{environ}
\RequirePackage{etoolbox}

%%======== Metadata ========
\newcommand{\subtitle}[1]{\gdef\@subtitle{#1}}
\newcommand{\shorttitle}[1]{\gdef\@shorttitle{#1}}
\newcommand{\venue}[1]{\gdef\@venue{#1}}
\newcommand{\submissionID}[1]{\gdef\@submissionID{#1}}
\subtitle{Extended Abstract}
\let\@shorttitle\@empty
\let\@venue\@empty
\submissionID{<<submission id>>}

\AtBeginDocument{%
  \ifx\@shorttitle\@empty\global\let\@shorttitle\@title\fi
}

%%======== Fonts for the new elements ========
\newcommand{\subtitlefont}{\normalfont\fontsize{12}{15}\selectfont}

%%======== Running head (pages 2+) ========
\fancyhead[C]{\footerfont \@shorttitle}

%%======== First-page header: configurable logo ========
% \logo[includegraphics options]{file}; default is the NVIDIA logo.
% The base class hard-codes its logo, so the first-page style is
% redefined here with the same header and footer content.
\newcommand{\logo}[2][width=90pt]{\gdef\@logo{\includegraphics[#1]{#2}}}
\logo{assets/nvlogo2.png}

\fancypagestyle{firststyle}{%
  \fancyhead[L]{\ifthenelse{\boolean{logo}}{\@logo}{}}%
  \fancyhead[C]{}%
  \fancyhead[R]{%
    \ifdefined\paperurl
      \if\relax\the\paperurl\relax\else
        \href{\the\paperurl}{\urlheaderfont\itshape\the\paperurl}\\
      \fi
    \fi
    {\footerfont\itshape\monthyeardate\today}}%
  \fancyfoot[L]{%
    \ifdefined\correspondingauthor
      \if\relax\the\correspondingauthor\relax\else
        \footerfont{* Equal contribution. \\}
      \fi
    \fi
    \ifthenelse{\boolean{internal}}{\footerfont\internalonly\\}{}%
    \ifthenelse{\boolean{address}}{%
      \itshape\footerfont 2788 San Tomas Expressway, Santa Clara, CA 95051, USA\\}{}%
    \copyrightext}%
  \fancyfoot[C]{}%
  \fancyfoot[R]{%
    \ifthenelse{\boolean{internal}}{%
      \ifdefined\reportnumber
        \if\relax\the\reportnumber\relax\else
          \footerfont\bfseries NVIDIA Technical Report \the\reportnumber
        \fi
      \fi}{}}%
}

%%======== First-page footer: venue line above the copyright ========
\renewcommand{\copyrightext}{\footerfont
  \ifx\@venue\@empty\else
    \textit{\@venue}\if@nvcopyright\\\fi
  \fi
  \if@nvcopyright\textcopyright\, \the\year{} NVIDIA. All rights reserved.\fi}

%%======== Title block ========
\renewcommand{\maketitle}{%
  \twocolumn[%
    \begin{adjustwidth}{0pt}{24pt}
      \begin{flushleft}
        {\raggedright\titlefont \@title\par}%
        \ifx\@subtitle\@empty\else
          \vskip4pt
          {\raggedright\subtitlefont \@subtitle\par}%
        \fi
        \vskip11pt
        \if@anonymous
          {\raggedright\Authfont Submission ID: \@submissionID\par}%
        \else
          {\raggedright \@author\par}%
        \fi
        \vskip16pt
      \end{flushleft}
    \end{adjustwidth}
    \abscontent
    \vskip16pt
  ]%
  \thispagestyle{firststyle}%
}

%%======== Page-limit check ========
% The main text is limited to 2 pages; references, acknowledgments and
% appendices do not count. Check the page number where the first of
% those starts.
\newif\if@bodychecked
\newcommand{\@checkbodylength}{%
  \if@bodychecked\else
    \global\@bodycheckedtrue
    \ifnum\value{page}>2\relax
      \ClassWarningNoLine{nvidiaextabstract}{%
        The main text appears to run onto page \the\value{page}.\MessageBreak
        It is limited to 2 pages, not counting references,\MessageBreak
        acknowledgments and appendices}%
    \fi
  \fi
}
\AtBeginEnvironment{thebibliography}{\@checkbodylength}
\pretocmd{\appendix}{\@checkbodylength}{}{}

%%======== Acknowledgments ========
\NewEnviron{acks}{%
  \@checkbodylength
  \if@anonymous\else
    \section*{Acknowledgments}\BODY
  \fi
}

%%======== Figure descriptions (accessibility text, not printed) ========
\newcommand{\Description}[2][]{}

%%======== Compact bibliography ========
\def\bibfont{\footnotesize}
\setlength{\bibsep}{1pt plus 0.5pt}

\endinput
