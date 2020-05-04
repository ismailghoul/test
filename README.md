# Multiclass Classification using Keras and TensorFlow on PlantVillage Dataset
<!DOCTYPE html>
<html class="gr__localhost" lang="en-us"><head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8">
    <meta charset="utf-8">

    <title>z - Jupyter Notebook</title>
    
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <link rel="stylesheet" href="test%20-%20Jupyter%20Notebook_files/jquery-ui.css" type="text/css">
    <link rel="stylesheet" href="test%20-%20Jupyter%20Notebook_files/jquery.css" type="text/css">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    


<script type="text/javascript" src="test%20-%20Jupyter%20Notebook_files/MathJax.js" charset="utf-8"></script>

<script type="text/javascript">
// MathJax disabled, set as null to distinguish from *missing* MathJax,
// where it will be undefined, and should prompt a dialog later.
window.mathjax_url = "/static/components/MathJax/MathJax.js";
</script>

<link rel="stylesheet" href="test%20-%20Jupyter%20Notebook_files/bootstrap-tour.css" type="text/css">
<link rel="stylesheet" href="test%20-%20Jupyter%20Notebook_files/codemirror.css">


    <link rel="stylesheet" href="test%20-%20Jupyter%20Notebook_files/style.css" type="text/css">
    

<link rel="stylesheet" href="test%20-%20Jupyter%20Notebook_files/override.css" type="text/css">
<link rel="stylesheet" href="" id="kernel-css" type="text/css">


    <link rel="stylesheet" href="test%20-%20Jupyter%20Notebook_files/custom.css" type="text/css">
    <script src="test%20-%20Jupyter%20Notebook_files/promise.js" type="text/javascript" charset="utf-8"></script>
    <script src="test%20-%20Jupyter%20Notebook_files/react.js" type="text/javascript"></script>
    <script src="test%20-%20Jupyter%20Notebook_files/react-dom.js" type="text/javascript"></script>
    <script src="test%20-%20Jupyter%20Notebook_files/index.js" type="text/javascript"></script>
    <script src="test%20-%20Jupyter%20Notebook_files/require.js" type="text/javascript" charset="utf-8"></script>
    <script>
      require.config({
          
          urlArgs: "v=20200504125913",
          
          baseUrl: '/static/',
          paths: {
            'auth/js/main': 'auth/js/main.min',
            custom : '/custom',
            nbextensions : '/nbextensions',
            kernelspecs : '/kernelspecs',
            underscore : 'components/underscore/underscore-min',
            backbone : 'components/backbone/backbone-min',
            jed: 'components/jed/jed',
            jquery: 'components/jquery/jquery.min',
            json: 'components/requirejs-plugins/src/json',
            text: 'components/requirejs-text/text',
            bootstrap: 'components/bootstrap/dist/js/bootstrap.min',
            bootstraptour: 'components/bootstrap-tour/build/js/bootstrap-tour.min',
            'jquery-ui': 'components/jquery-ui/jquery-ui.min',
            moment: 'components/moment/min/moment-with-locales',
            codemirror: 'components/codemirror',
            termjs: 'components/xterm.js/xterm',
            typeahead: 'components/jquery-typeahead/dist/jquery.typeahead.min',
          },
          map: { // for backward compatibility
              "*": {
                  "jqueryui": "jquery-ui",
              }
          },
          shim: {
            typeahead: {
              deps: ["jquery"],
              exports: "typeahead"
            },
            underscore: {
              exports: '_'
            },
            backbone: {
              deps: ["underscore", "jquery"],
              exports: "Backbone"
            },
            bootstrap: {
              deps: ["jquery"],
              exports: "bootstrap"
            },
            bootstraptour: {
              deps: ["bootstrap"],
              exports: "Tour"
            },
            "jquery-ui": {
              deps: ["jquery"],
              exports: "$"
            }
          },
          waitSeconds: 30,
      });

      require.config({
          map: {
              '*':{
                'contents': 'services/contents',
              }
          }
      });

      // error-catching custom.js shim.
      define("custom", function (require, exports, module) {
          try {
              var custom = require('custom/custom');
              console.debug('loaded custom.js');
              return custom;
          } catch (e) {
              console.error("error loading custom.js", e);
              return {};
          }
      })

    document.nbjs_translations = {"domain": "nbjs", "locale_data": {"nbjs": {"": {"domain": "nbjs"}}}};
    document.documentElement.lang = navigator.language.toLowerCase();
    </script>

    
    

<script type="text/javascript" charset="utf-8" async="" data-requirecontext="_" data-requiremodule="services/contents" src="test%20-%20Jupyter%20Notebook_files/contents.js"></script><style type="text/css">.MathJax_Hover_Frame {border-radius: .25em; -webkit-border-radius: .25em; -moz-border-radius: .25em; -khtml-border-radius: .25em; box-shadow: 0px 0px 15px #83A; -webkit-box-shadow: 0px 0px 15px #83A; -moz-box-shadow: 0px 0px 15px #83A; -khtml-box-shadow: 0px 0px 15px #83A; border: 1px solid #A6D ! important; display: inline-block; position: absolute}
.MathJax_Menu_Button .MathJax_Hover_Arrow {position: absolute; cursor: pointer; display: inline-block; border: 2px solid #AAA; border-radius: 4px; -webkit-border-radius: 4px; -moz-border-radius: 4px; -khtml-border-radius: 4px; font-family: 'Courier New',Courier; font-size: 9px; color: #F0F0F0}
.MathJax_Menu_Button .MathJax_Hover_Arrow span {display: block; background-color: #AAA; border: 1px solid; border-radius: 3px; line-height: 0; padding: 4px}
.MathJax_Hover_Arrow:hover {color: white!important; border: 2px solid #CCC!important}
.MathJax_Hover_Arrow:hover span {background-color: #CCC!important}
</style><style type="text/css">#MathJax_About {position: fixed; left: 50%; width: auto; text-align: center; border: 3px outset; padding: 1em 2em; background-color: #DDDDDD; color: black; cursor: default; font-family: message-box; font-size: 120%; font-style: normal; text-indent: 0; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; z-index: 201; border-radius: 15px; -webkit-border-radius: 15px; -moz-border-radius: 15px; -khtml-border-radius: 15px; box-shadow: 0px 10px 20px #808080; -webkit-box-shadow: 0px 10px 20px #808080; -moz-box-shadow: 0px 10px 20px #808080; -khtml-box-shadow: 0px 10px 20px #808080; filter: progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color='gray', Positive='true')}
#MathJax_About.MathJax_MousePost {outline: none}
.MathJax_Menu {position: absolute; background-color: white; color: black; width: auto; padding: 2px; border: 1px solid #CCCCCC; margin: 0; cursor: default; font: menu; text-align: left; text-indent: 0; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; z-index: 201; box-shadow: 0px 10px 20px #808080; -webkit-box-shadow: 0px 10px 20px #808080; -moz-box-shadow: 0px 10px 20px #808080; -khtml-box-shadow: 0px 10px 20px #808080; filter: progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color='gray', Positive='true')}
.MathJax_MenuItem {padding: 2px 2em; background: transparent}
.MathJax_MenuArrow {position: absolute; right: .5em; padding-top: .25em; color: #666666; font-size: .75em}
.MathJax_MenuActive .MathJax_MenuArrow {color: white}
.MathJax_MenuArrow.RTL {left: .5em; right: auto}
.MathJax_MenuCheck {position: absolute; left: .7em}
.MathJax_MenuCheck.RTL {right: .7em; left: auto}
.MathJax_MenuRadioCheck {position: absolute; left: 1em}
.MathJax_MenuRadioCheck.RTL {right: 1em; left: auto}
.MathJax_MenuLabel {padding: 2px 2em 4px 1.33em; font-style: italic}
.MathJax_MenuRule {border-top: 1px solid #CCCCCC; margin: 4px 1px 0px}
.MathJax_MenuDisabled {color: GrayText}
.MathJax_MenuActive {background-color: Highlight; color: HighlightText}
.MathJax_MenuDisabled:focus, .MathJax_MenuLabel:focus {background-color: #E8E8E8}
.MathJax_ContextMenu:focus {outline: none}
.MathJax_ContextMenu .MathJax_MenuItem:focus {outline: none}
#MathJax_AboutClose {top: .2em; right: .2em}
.MathJax_Menu .MathJax_MenuClose {top: -10px; left: -10px}
.MathJax_MenuClose {position: absolute; cursor: pointer; display: inline-block; border: 2px solid #AAA; border-radius: 18px; -webkit-border-radius: 18px; -moz-border-radius: 18px; -khtml-border-radius: 18px; font-family: 'Courier New',Courier; font-size: 24px; color: #F0F0F0}
.MathJax_MenuClose span {display: block; background-color: #AAA; border: 1.5px solid; border-radius: 18px; -webkit-border-radius: 18px; -moz-border-radius: 18px; -khtml-border-radius: 18px; line-height: 0; padding: 8px 0 6px}
.MathJax_MenuClose:hover {color: white!important; border: 2px solid #CCC!important}
.MathJax_MenuClose:hover span {background-color: #CCC!important}
.MathJax_MenuClose:hover:focus {outline: none}
</style><style type="text/css">.MathJax_Preview .MJXf-math {color: inherit!important}
</style><style type="text/css">.MJX_Assistive_MathML {position: absolute!important; top: 0; left: 0; clip: rect(1px, 1px, 1px, 1px); padding: 1px 0 0 0!important; border: 0!important; height: 1px!important; width: 1px!important; overflow: hidden!important; display: block!important; -webkit-touch-callout: none; -webkit-user-select: none; -khtml-user-select: none; -moz-user-select: none; -ms-user-select: none; user-select: none}
.MJX_Assistive_MathML.MJX_Assistive_MathML_Block {width: 100%!important}
</style><style type="text/css">#MathJax_Zoom {position: absolute; background-color: #F0F0F0; overflow: auto; display: block; z-index: 301; padding: .5em; border: 1px solid black; margin: 0; font-weight: normal; font-style: normal; text-align: left; text-indent: 0; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; -webkit-box-sizing: content-box; -moz-box-sizing: content-box; box-sizing: content-box; box-shadow: 5px 5px 15px #AAAAAA; -webkit-box-shadow: 5px 5px 15px #AAAAAA; -moz-box-shadow: 5px 5px 15px #AAAAAA; -khtml-box-shadow: 5px 5px 15px #AAAAAA; filter: progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color='gray', Positive='true')}
#MathJax_ZoomOverlay {position: absolute; left: 0; top: 0; z-index: 300; display: inline-block; width: 100%; height: 100%; border: 0; padding: 0; margin: 0; background-color: white; opacity: 0; filter: alpha(opacity=0)}
#MathJax_ZoomFrame {position: relative; display: inline-block; height: 0; width: 0}
#MathJax_ZoomEventTrap {position: absolute; left: 0; top: 0; z-index: 302; display: inline-block; border: 0; padding: 0; margin: 0; background-color: white; opacity: 0; filter: alpha(opacity=0)}
</style><style type="text/css">.MathJax_Preview {color: #888}
#MathJax_Message {position: fixed; left: 1px; bottom: 2px; background-color: #E6E6E6; border: 1px solid #959595; margin: 0px; padding: 2px 8px; z-index: 102; color: black; font-size: 80%; width: auto; white-space: nowrap}
#MathJax_MSIE_Frame {position: absolute; top: 0; left: 0; width: 0px; z-index: 101; border: 0px; margin: 0px; padding: 0px}
.MathJax_Error {color: #CC0000; font-style: italic}
</style><script type="text/javascript" charset="utf-8" async="" data-requirecontext="_" data-requiremodule="custom/custom" src="test%20-%20Jupyter%20Notebook_files/custom.js"></script><script type="text/javascript" charset="utf-8" async="" data-requirecontext="_" data-requiremodule="nbextensions/jupyter-js-widgets/extension" src="test%20-%20Jupyter%20Notebook_files/extension.js"></script><style type="text/css">/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/* Override the correction for the prompt area in https://github.com/jupyter/notebook/blob/dd41d9fd5c4f698bd7468612d877828a7eeb0e7a/IPython/html/static/notebook/less/outputarea.less#L110 */
.jupyter-widgets-output-area div.output_subarea {
    max-width: 100%;
}

/* Work-around for the bug fixed in https://github.com/jupyter/notebook/pull/2961 */
.jupyter-widgets-output-area > .out_prompt_overlay {
    display: none;
}
</style><style type="text/css">div.MathJax_MathML {text-align: center; margin: .75em 0px; display: block!important}
.MathJax_MathML {font-style: normal; font-weight: normal; line-height: normal; font-size: 100%; font-size-adjust: none; text-indent: 0; text-align: left; text-transform: none; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; direction: ltr; max-width: none; max-height: none; min-width: 0; min-height: 0; border: 0; padding: 0; margin: 0}
span.MathJax_MathML {display: inline!important}
.MathJax_mmlExBox {display: block!important; overflow: hidden; height: 1px; width: 60ex; min-height: 0; max-height: none; padding: 0; border: 0; margin: 0}
[class="MJX-tex-oldstyle"] {font-family: MathJax_Caligraphic, MathJax_Caligraphic-WEB}
[class="MJX-tex-oldstyle-bold"] {font-family: MathJax_Caligraphic, MathJax_Caligraphic-WEB; font-weight: bold}
[class="MJX-tex-caligraphic"] {font-family: MathJax_Caligraphic, MathJax_Caligraphic-WEB}
[class="MJX-tex-caligraphic-bold"] {font-family: MathJax_Caligraphic, MathJax_Caligraphic-WEB; font-weight: bold}
@font-face /*1*/ {font-family: MathJax_Caligraphic-WEB; src: url('http://localhost:8888/static/components/MathJax/fonts/HTML-CSS/TeX/otf/MathJax_Caligraphic-Regular.otf')}
@font-face /*2*/ {font-family: MathJax_Caligraphic-WEB; font-weight: bold; src: url('http://localhost:8888/static/components/MathJax/fonts/HTML-CSS/TeX/otf/MathJax_Caligraphic-Bold.otf')}
[mathvariant="double-struck"] {font-family: MathJax_AMS, MathJax_AMS-WEB}
[mathvariant="script"] {font-family: MathJax_Script, MathJax_Script-WEB}
[mathvariant="fraktur"] {font-family: MathJax_Fraktur, MathJax_Fraktur-WEB}
[mathvariant="bold-script"] {font-family: MathJax_Script, MathJax_Caligraphic-WEB; font-weight: bold}
[mathvariant="bold-fraktur"] {font-family: MathJax_Fraktur, MathJax_Fraktur-WEB; font-weight: bold}
[mathvariant="monospace"] {font-family: monospace}
[mathvariant="sans-serif"] {font-family: sans-serif}
[mathvariant="bold-sans-serif"] {font-family: sans-serif; font-weight: bold}
[mathvariant="sans-serif-italic"] {font-family: sans-serif; font-style: italic}
[mathvariant="sans-serif-bold-italic"] {font-family: sans-serif; font-style: italic; font-weight: bold}
@font-face /*3*/ {font-family: MathJax_AMS-WEB; src: url('http://localhost:8888/static/components/MathJax/fonts/HTML-CSS/TeX/otf/MathJax_AMS-Regular.otf')}
@font-face /*4*/ {font-family: MathJax_Script-WEB; src: url('http://localhost:8888/static/components/MathJax/fonts/HTML-CSS/TeX/otf/MathJax_Script-Regular.otf')}
@font-face /*5*/ {font-family: MathJax_Fraktur-WEB; src: url('http://localhost:8888/static/components/MathJax/fonts/HTML-CSS/TeX/otf/MathJax_Fraktur-Regular.otf')}
@font-face /*6*/ {font-family: MathJax_Fraktur-WEB; font-weight: bold; src: url('http://localhost:8888/static/components/MathJax/fonts/HTML-CSS/TeX/otf/MathJax_Fraktur-Bold.otf')}
</style><style type="text/css">.MathJax_Display {text-align: center; margin: 0; position: relative; display: block!important; text-indent: 0; max-width: none; max-height: none; min-width: 0; min-height: 0; width: 100%}
.MathJax .merror {background-color: #FFFF88; color: #CC0000; border: 1px solid #CC0000; padding: 1px 3px; font-style: normal; font-size: 90%}
.MathJax .MJX-monospace {font-family: monospace}
.MathJax .MJX-sans-serif {font-family: sans-serif}
#MathJax_Tooltip {background-color: InfoBackground; color: InfoText; border: 1px solid black; box-shadow: 2px 2px 5px #AAAAAA; -webkit-box-shadow: 2px 2px 5px #AAAAAA; -moz-box-shadow: 2px 2px 5px #AAAAAA; -khtml-box-shadow: 2px 2px 5px #AAAAAA; filter: progid:DXImageTransform.Microsoft.dropshadow(OffX=2, OffY=2, Color='gray', Positive='true'); padding: 3px 4px; z-index: 401; position: absolute; left: 0; top: 0; width: auto; height: auto; display: none}
.MathJax {display: inline; font-style: normal; font-weight: normal; line-height: normal; font-size: 100%; font-size-adjust: none; text-indent: 0; text-align: left; text-transform: none; letter-spacing: normal; word-spacing: normal; word-wrap: normal; white-space: nowrap; float: none; direction: ltr; max-width: none; max-height: none; min-width: 0; min-height: 0; border: 0; padding: 0; margin: 0}
.MathJax:focus, body :focus .MathJax {display: inline-table}
.MathJax.MathJax_FullWidth {text-align: center; display: table-cell!important; width: 10000em!important}
.MathJax img, .MathJax nobr, .MathJax a {border: 0; padding: 0; margin: 0; max-width: none; max-height: none; min-width: 0; min-height: 0; vertical-align: 0; line-height: normal; text-decoration: none}
img.MathJax_strut {border: 0!important; padding: 0!important; margin: 0!important; vertical-align: 0!important}
.MathJax span {display: inline; position: static; border: 0; padding: 0; margin: 0; vertical-align: 0; line-height: normal; text-decoration: none; box-sizing: content-box}
.MathJax nobr {white-space: nowrap!important}
.MathJax img {display: inline!important; float: none!important}
.MathJax * {transition: none; -webkit-transition: none; -moz-transition: none; -ms-transition: none; -o-transition: none}
.MathJax_Processing {visibility: hidden; position: fixed; width: 0; height: 0; overflow: hidden}
.MathJax_Processed {display: none!important}
.MathJax_test {font-style: normal; font-weight: normal; font-size: 100%; font-size-adjust: none; text-indent: 0; text-transform: none; letter-spacing: normal; word-spacing: normal; overflow: hidden; height: 1px}
.MathJax_test.mjx-test-display {display: table!important}
.MathJax_test.mjx-test-inline {display: inline!important; margin-right: -1px}
.MathJax_test.mjx-test-default {display: block!important; clear: both}
.MathJax_ex_box {display: inline-block!important; position: absolute; overflow: hidden; min-height: 0; max-height: none; padding: 0; border: 0; margin: 0; width: 1px; height: 60ex}
.MathJax_em_box {display: inline-block!important; position: absolute; overflow: hidden; min-height: 0; max-height: none; padding: 0; border: 0; margin: 0; width: 1px; height: 60em}
.mjx-test-inline .MathJax_left_box {display: inline-block; width: 0; float: left}
.mjx-test-inline .MathJax_right_box {display: inline-block; width: 0; float: right}
.mjx-test-display .MathJax_right_box {display: table-cell!important; width: 10000em!important; min-width: 0; max-width: none; padding: 0; border: 0; margin: 0}
.MathJax .MathJax_HitBox {cursor: text; background: white; opacity: 0; filter: alpha(opacity=0)}
.MathJax .MathJax_HitBox * {filter: none; opacity: 1; background: transparent}
#MathJax_Tooltip * {filter: none; opacity: 1; background: transparent}
@font-face {font-family: MathJax_Blank; src: url('about:blank')}
.MathJax .noError {vertical-align: ; font-size: 90%; text-align: left; color: black; padding: 1px 3px; border: 1px solid}
</style><style type="text/css">.MJXp-script {font-size: .8em}
.MJXp-right {-webkit-transform-origin: right; -moz-transform-origin: right; -ms-transform-origin: right; -o-transform-origin: right; transform-origin: right}
.MJXp-bold {font-weight: bold}
.MJXp-italic {font-style: italic}
.MJXp-scr {font-family: MathJax_Script,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-frak {font-family: MathJax_Fraktur,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-sf {font-family: MathJax_SansSerif,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-cal {font-family: MathJax_Caligraphic,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-mono {font-family: MathJax_Typewriter,'Times New Roman',Times,STIXGeneral,serif}
.MJXp-largeop {font-size: 150%}
.MJXp-largeop.MJXp-int {vertical-align: -.2em}
.MJXp-math {display: inline-block; line-height: 1.2; text-indent: 0; font-family: 'Times New Roman',Times,STIXGeneral,serif; white-space: nowrap; border-collapse: collapse}
.MJXp-display {display: block; text-align: center; margin: 1em 0}
.MJXp-math span {display: inline-block}
.MJXp-box {display: block!important; text-align: center}
.MJXp-box:after {content: " "}
.MJXp-rule {display: block!important; margin-top: .1em}
.MJXp-char {display: block!important}
.MJXp-mo {margin: 0 .15em}
.MJXp-mfrac {margin: 0 .125em; vertical-align: .25em}
.MJXp-denom {display: inline-table!important; width: 100%}
.MJXp-denom > * {display: table-row!important}
.MJXp-surd {vertical-align: top}
.MJXp-surd > * {display: block!important}
.MJXp-script-box > *  {display: table!important; height: 50%}
.MJXp-script-box > * > * {display: table-cell!important; vertical-align: top}
.MJXp-script-box > *:last-child > * {vertical-align: bottom}
.MJXp-script-box > * > * > * {display: block!important}
.MJXp-mphantom {visibility: hidden}
.MJXp-munderover, .MJXp-munder {display: inline-table!important}
.MJXp-over {display: inline-block!important; text-align: center}
.MJXp-over > * {display: block!important}
.MJXp-munderover > *, .MJXp-munder > * {display: table-row!important}
.MJXp-mtable {vertical-align: .25em; margin: 0 .125em}
.MJXp-mtable > * {display: inline-table!important; vertical-align: middle}
.MJXp-mtr {display: table-row!important}
.MJXp-mtd {display: table-cell!important; text-align: center; padding: .5em 0 0 .5em}
.MJXp-mtr > .MJXp-mtd:first-child {padding-left: 0}
.MJXp-mtr:first-child > .MJXp-mtd {padding-top: 0}
.MJXp-mlabeledtr {display: table-row!important}
.MJXp-mlabeledtr > .MJXp-mtd:first-child {padding-left: 0}
.MJXp-mlabeledtr:first-child > .MJXp-mtd {padding-top: 0}
.MJXp-merror {background-color: #FFFF88; color: #CC0000; border: 1px solid #CC0000; padding: 1px 3px; font-style: normal; font-size: 90%}
.MJXp-scale0 {-webkit-transform: scaleX(.0); -moz-transform: scaleX(.0); -ms-transform: scaleX(.0); -o-transform: scaleX(.0); transform: scaleX(.0)}
.MJXp-scale1 {-webkit-transform: scaleX(.1); -moz-transform: scaleX(.1); -ms-transform: scaleX(.1); -o-transform: scaleX(.1); transform: scaleX(.1)}
.MJXp-scale2 {-webkit-transform: scaleX(.2); -moz-transform: scaleX(.2); -ms-transform: scaleX(.2); -o-transform: scaleX(.2); transform: scaleX(.2)}
.MJXp-scale3 {-webkit-transform: scaleX(.3); -moz-transform: scaleX(.3); -ms-transform: scaleX(.3); -o-transform: scaleX(.3); transform: scaleX(.3)}
.MJXp-scale4 {-webkit-transform: scaleX(.4); -moz-transform: scaleX(.4); -ms-transform: scaleX(.4); -o-transform: scaleX(.4); transform: scaleX(.4)}
.MJXp-scale5 {-webkit-transform: scaleX(.5); -moz-transform: scaleX(.5); -ms-transform: scaleX(.5); -o-transform: scaleX(.5); transform: scaleX(.5)}
.MJXp-scale6 {-webkit-transform: scaleX(.6); -moz-transform: scaleX(.6); -ms-transform: scaleX(.6); -o-transform: scaleX(.6); transform: scaleX(.6)}
.MJXp-scale7 {-webkit-transform: scaleX(.7); -moz-transform: scaleX(.7); -ms-transform: scaleX(.7); -o-transform: scaleX(.7); transform: scaleX(.7)}
.MJXp-scale8 {-webkit-transform: scaleX(.8); -moz-transform: scaleX(.8); -ms-transform: scaleX(.8); -o-transform: scaleX(.8); transform: scaleX(.8)}
.MJXp-scale9 {-webkit-transform: scaleX(.9); -moz-transform: scaleX(.9); -ms-transform: scaleX(.9); -o-transform: scaleX(.9); transform: scaleX(.9)}
.MathJax_PHTML .noError {vertical-align: ; font-size: 90%; text-align: left; color: black; padding: 1px 3px; border: 1px solid}
</style><style type="text/css">/*-----------------------------------------------------------------------------

| Copyright (c) 2014-2017, PhosphorJS Contributors

|

| Distributed under the terms of the BSD 3-Clause License.

|

| The full license is in the file LICENSE, distributed with this software.

|----------------------------------------------------------------------------*/





.p-Widget {

  box-sizing: border-box;

  position: relative;

  overflow: hidden;

  cursor: default;

}





.p-Widget.p-mod-hidden {

  display: none !important;

}

</style><style type="text/css">/*-----------------------------------------------------------------------------

| Copyright (c) 2014-2017, PhosphorJS Contributors

|

| Distributed under the terms of the BSD 3-Clause License.

|

| The full license is in the file LICENSE, distributed with this software.

|----------------------------------------------------------------------------*/





.p-CommandPalette {

  display: flex;

  flex-direction: column;

  -webkit-user-select: none;

  -moz-user-select: none;

  -ms-user-select: none;

  user-select: none;

}





.p-CommandPalette-search {

  flex: 0 0 auto;

}





.p-CommandPalette-content {

  flex: 1 1 auto;

  margin: 0;

  padding: 0;

  min-height: 0;

  overflow: auto;

  list-style-type: none;

}





.p-CommandPalette-header {

  overflow: hidden;

  white-space: nowrap;

  text-overflow: ellipsis;

}





.p-CommandPalette-item {

  display: flex;

  flex-direction: row;

}





.p-CommandPalette-itemIcon {

  flex: 0 0 auto;

}





.p-CommandPalette-itemContent {

  flex: 1 1 auto;

  overflow: hidden;

}





.p-CommandPalette-itemShortcut {

  flex: 0 0 auto;

}





.p-CommandPalette-itemLabel {

  overflow: hidden;

  white-space: nowrap;

  text-overflow: ellipsis;

}

</style><style type="text/css">/*-----------------------------------------------------------------------------

| Copyright (c) 2014-2017, PhosphorJS Contributors

|

| Distributed under the terms of the BSD 3-Clause License.

|

| The full license is in the file LICENSE, distributed with this software.

|----------------------------------------------------------------------------*/





.p-DockPanel {

  z-index: 0;

}





.p-DockPanel-widget {

  z-index: 0;

}





.p-DockPanel-tabBar {

  z-index: 1;

}





.p-DockPanel-handle {

  z-index: 2;

}





.p-DockPanel-handle.p-mod-hidden {

  display: none !important;

}





.p-DockPanel-handle:after {

  position: absolute;

  top: 0;

  left: 0;

  width: 100%;

  height: 100%;

  content: '';

}





.p-DockPanel-handle[data-orientation='horizontal'] {

  cursor: ew-resize;

}





.p-DockPanel-handle[data-orientation='vertical'] {

  cursor: ns-resize;

}





.p-DockPanel-handle[data-orientation='horizontal']:after {

  left: 50%;

  min-width: 8px;

  transform: translateX(-50%);

}





.p-DockPanel-handle[data-orientation='vertical']:after {

  top: 50%;

  min-height: 8px;

  transform: translateY(-50%);

}





.p-DockPanel-overlay {

  z-index: 3;

  box-sizing: border-box;

  pointer-events: none;

}





.p-DockPanel-overlay.p-mod-hidden {

  display: none !important;

}

</style><style type="text/css">/*-----------------------------------------------------------------------------

| Copyright (c) 2014-2017, PhosphorJS Contributors

|

| Distributed under the terms of the BSD 3-Clause License.

|

| The full license is in the file LICENSE, distributed with this software.

|----------------------------------------------------------------------------*/





.p-Menu {

  z-index: 10000;

  position: absolute;

  white-space: nowrap;

  overflow-x: hidden;

  overflow-y: auto;

  outline: none;

  -webkit-user-select: none;

  -moz-user-select: none;

  -ms-user-select: none;

  user-select: none;

}





.p-Menu-content {

  margin: 0;

  padding: 0;

  display: table;

  list-style-type: none;

}





.p-Menu-item {

  display: table-row;

}





.p-Menu-item.p-mod-hidden,

.p-Menu-item.p-mod-collapsed {

  display: none !important;

}





.p-Menu-itemIcon,

.p-Menu-itemSubmenuIcon {

  display: table-cell;

  text-align: center;

}





.p-Menu-itemLabel {

  display: table-cell;

  text-align: left;

}





.p-Menu-itemShortcut {

  display: table-cell;

  text-align: right;

}

</style><style type="text/css">/*-----------------------------------------------------------------------------

| Copyright (c) 2014-2017, PhosphorJS Contributors

|

| Distributed under the terms of the BSD 3-Clause License.

|

| The full license is in the file LICENSE, distributed with this software.

|----------------------------------------------------------------------------*/





.p-MenuBar {

  outline: none;

  -webkit-user-select: none;

  -moz-user-select: none;

  -ms-user-select: none;

  user-select: none;

}





.p-MenuBar-content {

  margin: 0;

  padding: 0;

  display: flex;

  flex-direction: row;

  list-style-type: none;

}





.p-MenuBar-item {

  box-sizing: border-box;

}





.p-MenuBar-itemIcon,

.p-MenuBar-itemLabel {

  display: inline-block;

}

</style><style type="text/css">/*-----------------------------------------------------------------------------

| Copyright (c) 2014-2017, PhosphorJS Contributors

|

| Distributed under the terms of the BSD 3-Clause License.

|

| The full license is in the file LICENSE, distributed with this software.

|----------------------------------------------------------------------------*/





.p-ScrollBar {

  display: flex;

  -webkit-user-select: none;

  -moz-user-select: none;

  -ms-user-select: none;

  user-select: none;

}





.p-ScrollBar[data-orientation='horizontal'] {

  flex-direction: row;

}





.p-ScrollBar[data-orientation='vertical'] {

  flex-direction: column;

}





.p-ScrollBar-button {

  box-sizing: border-box;

  flex: 0 0 auto;

}





.p-ScrollBar-track {

  box-sizing: border-box;

  position: relative;

  overflow: hidden;

  flex: 1 1 auto;

}





.p-ScrollBar-thumb {

  box-sizing: border-box;

  position: absolute;

}

</style><style type="text/css">/*-----------------------------------------------------------------------------

| Copyright (c) 2014-2017, PhosphorJS Contributors

|

| Distributed under the terms of the BSD 3-Clause License.

|

| The full license is in the file LICENSE, distributed with this software.

|----------------------------------------------------------------------------*/





.p-SplitPanel-child {

  z-index: 0;

}





.p-SplitPanel-handle {

  z-index: 1;

}





.p-SplitPanel-handle.p-mod-hidden {

  display: none !important;

}





.p-SplitPanel-handle:after {

  position: absolute;

  top: 0;

  left: 0;

  width: 100%;

  height: 100%;

  content: '';

}





.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle {

  cursor: ew-resize;

}





.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle {

  cursor: ns-resize;

}





.p-SplitPanel[data-orientation='horizontal'] > .p-SplitPanel-handle:after {

  left: 50%;

  min-width: 8px;

  transform: translateX(-50%);

}





.p-SplitPanel[data-orientation='vertical'] > .p-SplitPanel-handle:after {

  top: 50%;

  min-height: 8px;

  transform: translateY(-50%);

}

</style><style type="text/css">/*-----------------------------------------------------------------------------

| Copyright (c) 2014-2017, PhosphorJS Contributors

|

| Distributed under the terms of the BSD 3-Clause License.

|

| The full license is in the file LICENSE, distributed with this software.

|----------------------------------------------------------------------------*/





.p-TabBar {

  display: flex;

  -webkit-user-select: none;

  -moz-user-select: none;

  -ms-user-select: none;

  user-select: none;

}





.p-TabBar[data-orientation='horizontal'] {

  flex-direction: row;

}





.p-TabBar[data-orientation='vertical'] {

  flex-direction: column;

}





.p-TabBar-content {

  margin: 0;

  padding: 0;

  display: flex;

  flex: 1 1 auto;

  list-style-type: none;

}





.p-TabBar[data-orientation='horizontal'] > .p-TabBar-content {

  flex-direction: row;

}





.p-TabBar[data-orientation='vertical'] > .p-TabBar-content {

  flex-direction: column;

}





.p-TabBar-tab {

  display: flex;

  flex-direction: row;

  box-sizing: border-box;

  overflow: hidden;

}





.p-TabBar-tabIcon,

.p-TabBar-tabCloseIcon {

  flex: 0 0 auto;

}





.p-TabBar-tabLabel {

  flex: 1 1 auto;

  overflow: hidden;

  white-space: nowrap;

}





.p-TabBar-tab.p-mod-hidden {

  display: none !important;

}





.p-TabBar.p-mod-dragging .p-TabBar-tab {

  position: relative;

}





.p-TabBar.p-mod-dragging[data-orientation='horizontal'] .p-TabBar-tab {

  left: 0;

  transition: left 150ms ease;

}





.p-TabBar.p-mod-dragging[data-orientation='vertical'] .p-TabBar-tab {

  top: 0;

  transition: top 150ms ease;

}





.p-TabBar.p-mod-dragging .p-TabBar-tab.p-mod-dragging {

  transition: none;

}

</style><style type="text/css">/*-----------------------------------------------------------------------------

| Copyright (c) 2014-2017, PhosphorJS Contributors

|

| Distributed under the terms of the BSD 3-Clause License.

|

| The full license is in the file LICENSE, distributed with this software.

|----------------------------------------------------------------------------*/





.p-TabPanel-tabBar {

  z-index: 1;

}





.p-TabPanel-stackedPanel {

  z-index: 0;

}

</style><style type="text/css">/*-----------------------------------------------------------------------------

| Copyright (c) 2014-2017, PhosphorJS Contributors

|

| Distributed under the terms of the BSD 3-Clause License.

|

| The full license is in the file LICENSE, distributed with this software.

|----------------------------------------------------------------------------*/

</style><style type="text/css">/* Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

 .jupyter-widgets-disconnected::before {
    content: "\f127"; /* chain-broken */
    display: inline-block;
    font: normal normal normal 14px/1 FontAwesome;
    font-size: inherit;
    text-rendering: auto;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    color: #d9534f;
    padding: 3px;
    align-self: flex-start;
}
</style><style type="text/css">/**
 * The material design colors are adapted from google-material-color v1.2.6
 * https://github.com/danlevan/google-material-color
 * https://github.com/danlevan/google-material-color/blob/f67ca5f4028b2f1b34862f64b0ca67323f91b088/dist/palette.var.css
 *
 * The license for the material design color CSS variables is as follows (see
 * https://github.com/danlevan/google-material-color/blob/f67ca5f4028b2f1b34862f64b0ca67323f91b088/LICENSE)
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2014 Dan Le Van
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
:root {
  --md-red-50: #FFEBEE;
  --md-red-100: #FFCDD2;
  --md-red-200: #EF9A9A;
  --md-red-300: #E57373;
  --md-red-400: #EF5350;
  --md-red-500: #F44336;
  --md-red-600: #E53935;
  --md-red-700: #D32F2F;
  --md-red-800: #C62828;
  --md-red-900: #B71C1C;
  --md-red-A100: #FF8A80;
  --md-red-A200: #FF5252;
  --md-red-A400: #FF1744;
  --md-red-A700: #D50000;

  --md-pink-50: #FCE4EC;
  --md-pink-100: #F8BBD0;
  --md-pink-200: #F48FB1;
  --md-pink-300: #F06292;
  --md-pink-400: #EC407A;
  --md-pink-500: #E91E63;
  --md-pink-600: #D81B60;
  --md-pink-700: #C2185B;
  --md-pink-800: #AD1457;
  --md-pink-900: #880E4F;
  --md-pink-A100: #FF80AB;
  --md-pink-A200: #FF4081;
  --md-pink-A400: #F50057;
  --md-pink-A700: #C51162;

  --md-purple-50: #F3E5F5;
  --md-purple-100: #E1BEE7;
  --md-purple-200: #CE93D8;
  --md-purple-300: #BA68C8;
  --md-purple-400: #AB47BC;
  --md-purple-500: #9C27B0;
  --md-purple-600: #8E24AA;
  --md-purple-700: #7B1FA2;
  --md-purple-800: #6A1B9A;
  --md-purple-900: #4A148C;
  --md-purple-A100: #EA80FC;
  --md-purple-A200: #E040FB;
  --md-purple-A400: #D500F9;
  --md-purple-A700: #AA00FF;

  --md-deep-purple-50: #EDE7F6;
  --md-deep-purple-100: #D1C4E9;
  --md-deep-purple-200: #B39DDB;
  --md-deep-purple-300: #9575CD;
  --md-deep-purple-400: #7E57C2;
  --md-deep-purple-500: #673AB7;
  --md-deep-purple-600: #5E35B1;
  --md-deep-purple-700: #512DA8;
  --md-deep-purple-800: #4527A0;
  --md-deep-purple-900: #311B92;
  --md-deep-purple-A100: #B388FF;
  --md-deep-purple-A200: #7C4DFF;
  --md-deep-purple-A400: #651FFF;
  --md-deep-purple-A700: #6200EA;

  --md-indigo-50: #E8EAF6;
  --md-indigo-100: #C5CAE9;
  --md-indigo-200: #9FA8DA;
  --md-indigo-300: #7986CB;
  --md-indigo-400: #5C6BC0;
  --md-indigo-500: #3F51B5;
  --md-indigo-600: #3949AB;
  --md-indigo-700: #303F9F;
  --md-indigo-800: #283593;
  --md-indigo-900: #1A237E;
  --md-indigo-A100: #8C9EFF;
  --md-indigo-A200: #536DFE;
  --md-indigo-A400: #3D5AFE;
  --md-indigo-A700: #304FFE;

  --md-blue-50: #E3F2FD;
  --md-blue-100: #BBDEFB;
  --md-blue-200: #90CAF9;
  --md-blue-300: #64B5F6;
  --md-blue-400: #42A5F5;
  --md-blue-500: #2196F3;
  --md-blue-600: #1E88E5;
  --md-blue-700: #1976D2;
  --md-blue-800: #1565C0;
  --md-blue-900: #0D47A1;
  --md-blue-A100: #82B1FF;
  --md-blue-A200: #448AFF;
  --md-blue-A400: #2979FF;
  --md-blue-A700: #2962FF;

  --md-light-blue-50: #E1F5FE;
  --md-light-blue-100: #B3E5FC;
  --md-light-blue-200: #81D4FA;
  --md-light-blue-300: #4FC3F7;
  --md-light-blue-400: #29B6F6;
  --md-light-blue-500: #03A9F4;
  --md-light-blue-600: #039BE5;
  --md-light-blue-700: #0288D1;
  --md-light-blue-800: #0277BD;
  --md-light-blue-900: #01579B;
  --md-light-blue-A100: #80D8FF;
  --md-light-blue-A200: #40C4FF;
  --md-light-blue-A400: #00B0FF;
  --md-light-blue-A700: #0091EA;

  --md-cyan-50: #E0F7FA;
  --md-cyan-100: #B2EBF2;
  --md-cyan-200: #80DEEA;
  --md-cyan-300: #4DD0E1;
  --md-cyan-400: #26C6DA;
  --md-cyan-500: #00BCD4;
  --md-cyan-600: #00ACC1;
  --md-cyan-700: #0097A7;
  --md-cyan-800: #00838F;
  --md-cyan-900: #006064;
  --md-cyan-A100: #84FFFF;
  --md-cyan-A200: #18FFFF;
  --md-cyan-A400: #00E5FF;
  --md-cyan-A700: #00B8D4;

  --md-teal-50: #E0F2F1;
  --md-teal-100: #B2DFDB;
  --md-teal-200: #80CBC4;
  --md-teal-300: #4DB6AC;
  --md-teal-400: #26A69A;
  --md-teal-500: #009688;
  --md-teal-600: #00897B;
  --md-teal-700: #00796B;
  --md-teal-800: #00695C;
  --md-teal-900: #004D40;
  --md-teal-A100: #A7FFEB;
  --md-teal-A200: #64FFDA;
  --md-teal-A400: #1DE9B6;
  --md-teal-A700: #00BFA5;

  --md-green-50: #E8F5E9;
  --md-green-100: #C8E6C9;
  --md-green-200: #A5D6A7;
  --md-green-300: #81C784;
  --md-green-400: #66BB6A;
  --md-green-500: #4CAF50;
  --md-green-600: #43A047;
  --md-green-700: #388E3C;
  --md-green-800: #2E7D32;
  --md-green-900: #1B5E20;
  --md-green-A100: #B9F6CA;
  --md-green-A200: #69F0AE;
  --md-green-A400: #00E676;
  --md-green-A700: #00C853;

  --md-light-green-50: #F1F8E9;
  --md-light-green-100: #DCEDC8;
  --md-light-green-200: #C5E1A5;
  --md-light-green-300: #AED581;
  --md-light-green-400: #9CCC65;
  --md-light-green-500: #8BC34A;
  --md-light-green-600: #7CB342;
  --md-light-green-700: #689F38;
  --md-light-green-800: #558B2F;
  --md-light-green-900: #33691E;
  --md-light-green-A100: #CCFF90;
  --md-light-green-A200: #B2FF59;
  --md-light-green-A400: #76FF03;
  --md-light-green-A700: #64DD17;

  --md-lime-50: #F9FBE7;
  --md-lime-100: #F0F4C3;
  --md-lime-200: #E6EE9C;
  --md-lime-300: #DCE775;
  --md-lime-400: #D4E157;
  --md-lime-500: #CDDC39;
  --md-lime-600: #C0CA33;
  --md-lime-700: #AFB42B;
  --md-lime-800: #9E9D24;
  --md-lime-900: #827717;
  --md-lime-A100: #F4FF81;
  --md-lime-A200: #EEFF41;
  --md-lime-A400: #C6FF00;
  --md-lime-A700: #AEEA00;

  --md-yellow-50: #FFFDE7;
  --md-yellow-100: #FFF9C4;
  --md-yellow-200: #FFF59D;
  --md-yellow-300: #FFF176;
  --md-yellow-400: #FFEE58;
  --md-yellow-500: #FFEB3B;
  --md-yellow-600: #FDD835;
  --md-yellow-700: #FBC02D;
  --md-yellow-800: #F9A825;
  --md-yellow-900: #F57F17;
  --md-yellow-A100: #FFFF8D;
  --md-yellow-A200: #FFFF00;
  --md-yellow-A400: #FFEA00;
  --md-yellow-A700: #FFD600;

  --md-amber-50: #FFF8E1;
  --md-amber-100: #FFECB3;
  --md-amber-200: #FFE082;
  --md-amber-300: #FFD54F;
  --md-amber-400: #FFCA28;
  --md-amber-500: #FFC107;
  --md-amber-600: #FFB300;
  --md-amber-700: #FFA000;
  --md-amber-800: #FF8F00;
  --md-amber-900: #FF6F00;
  --md-amber-A100: #FFE57F;
  --md-amber-A200: #FFD740;
  --md-amber-A400: #FFC400;
  --md-amber-A700: #FFAB00;

  --md-orange-50: #FFF3E0;
  --md-orange-100: #FFE0B2;
  --md-orange-200: #FFCC80;
  --md-orange-300: #FFB74D;
  --md-orange-400: #FFA726;
  --md-orange-500: #FF9800;
  --md-orange-600: #FB8C00;
  --md-orange-700: #F57C00;
  --md-orange-800: #EF6C00;
  --md-orange-900: #E65100;
  --md-orange-A100: #FFD180;
  --md-orange-A200: #FFAB40;
  --md-orange-A400: #FF9100;
  --md-orange-A700: #FF6D00;

  --md-deep-orange-50: #FBE9E7;
  --md-deep-orange-100: #FFCCBC;
  --md-deep-orange-200: #FFAB91;
  --md-deep-orange-300: #FF8A65;
  --md-deep-orange-400: #FF7043;
  --md-deep-orange-500: #FF5722;
  --md-deep-orange-600: #F4511E;
  --md-deep-orange-700: #E64A19;
  --md-deep-orange-800: #D84315;
  --md-deep-orange-900: #BF360C;
  --md-deep-orange-A100: #FF9E80;
  --md-deep-orange-A200: #FF6E40;
  --md-deep-orange-A400: #FF3D00;
  --md-deep-orange-A700: #DD2C00;

  --md-brown-50: #EFEBE9;
  --md-brown-100: #D7CCC8;
  --md-brown-200: #BCAAA4;
  --md-brown-300: #A1887F;
  --md-brown-400: #8D6E63;
  --md-brown-500: #795548;
  --md-brown-600: #6D4C41;
  --md-brown-700: #5D4037;
  --md-brown-800: #4E342E;
  --md-brown-900: #3E2723;

  --md-grey-50: #FAFAFA;
  --md-grey-100: #F5F5F5;
  --md-grey-200: #EEEEEE;
  --md-grey-300: #E0E0E0;
  --md-grey-400: #BDBDBD;
  --md-grey-500: #9E9E9E;
  --md-grey-600: #757575;
  --md-grey-700: #616161;
  --md-grey-800: #424242;
  --md-grey-900: #212121;

  --md-blue-grey-50: #ECEFF1;
  --md-blue-grey-100: #CFD8DC;
  --md-blue-grey-200: #B0BEC5;
  --md-blue-grey-300: #90A4AE;
  --md-blue-grey-400: #78909C;
  --md-blue-grey-500: #607D8B;
  --md-blue-grey-600: #546E7A;
  --md-blue-grey-700: #455A64;
  --md-blue-grey-800: #37474F;
  --md-blue-grey-900: #263238;
}</style><style type="text/css">/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
This file is copied from the JupyterLab project to define default styling for
when the widget styling is compiled down to eliminate CSS variables. We make one
change - we comment out the font import below.
*/

/*
The following CSS variables define the main, public API for styling JupyterLab.
These variables should be used by all plugins wherever possible. In other
words, plugins should not define custom colors, sizes, etc unless absolutely
necessary. This enables users to change the visual theme of JupyterLab
by changing these variables.

Many variables appear in an ordered sequence (0,1,2,3). These sequences
are designed to work well together, so for example, `--jp-border-color1` should
be used with `--jp-layout-color1`. The numbers have the following meanings:

* 0: super-primary, reserved for special emphasis
* 1: primary, most important under normal situations
* 2: secondary, next most important under normal situations
* 3: tertiary, next most important under normal situations

Throughout JupyterLab, we are mostly following principles from Google's
Material Design when selecting colors. We are not, however, following
all of MD as it is not optimized for dense, information rich UIs.
*/


/*
 * Optional monospace font for input/output prompt.
 */
 /* Commented out in ipywidgets since we don't need it. */
/* @import url('https://fonts.googleapis.com/css?family=Roboto+Mono'); */

/*
 * Added for compabitility with output area
 */
:root {
  --jp-icon-search: none;
  --jp-ui-select-caret: none;
}


:root {

  /* Borders

  The following variables, specify the visual styling of borders in JupyterLab.
   */

  --jp-border-width: 1px;
  --jp-border-color0: var(--md-grey-700);
  --jp-border-color1: var(--md-grey-500);
  --jp-border-color2: var(--md-grey-300);
  --jp-border-color3: var(--md-grey-100);

  /* UI Fonts

  The UI font CSS variables are used for the typography all of the JupyterLab
  user interface elements that are not directly user generated content.
  */

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: calc(var(--jp-ui-font-size1)/var(--jp-ui-font-scale-factor));
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: calc(var(--jp-ui-font-size1)*var(--jp-ui-font-scale-factor));
  --jp-ui-font-size3: calc(var(--jp-ui-font-size2)*var(--jp-ui-font-scale-factor));
  --jp-ui-icon-font-size: 14px; /* Ensures px perfect FontAwesome icons */
  --jp-ui-font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;

  /* Use these font colors against the corresponding main layout colors.
     In a light theme, these go from dark to light.
  */

  --jp-ui-font-color0: rgba(0,0,0,1.0);
  --jp-ui-font-color1: rgba(0,0,0,0.8);
  --jp-ui-font-color2: rgba(0,0,0,0.5);
  --jp-ui-font-color3: rgba(0,0,0,0.3);

  /* Use these against the brand/accent/warn/error colors.
     These will typically go from light to darker, in both a dark and light theme
   */

  --jp-inverse-ui-font-color0: rgba(255,255,255,1);
  --jp-inverse-ui-font-color1: rgba(255,255,255,1.0);
  --jp-inverse-ui-font-color2: rgba(255,255,255,0.7);
  --jp-inverse-ui-font-color3: rgba(255,255,255,0.5);

  /* Content Fonts

  Content font variables are used for typography of user generated content.
  */

  --jp-content-font-size: 13px;
  --jp-content-line-height: 1.5;
  --jp-content-font-color0: black;
  --jp-content-font-color1: black;
  --jp-content-font-color2: var(--md-grey-700);
  --jp-content-font-color3: var(--md-grey-500);

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: calc(var(--jp-ui-font-size1)/var(--jp-ui-font-scale-factor));
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: calc(var(--jp-ui-font-size1)*var(--jp-ui-font-scale-factor));
  --jp-ui-font-size3: calc(var(--jp-ui-font-size2)*var(--jp-ui-font-scale-factor));

  --jp-code-font-size: 13px;
  --jp-code-line-height: 1.307;
  --jp-code-padding: 5px;
  --jp-code-font-family: monospace;


  /* Layout

  The following are the main layout colors use in JupyterLab. In a light
  theme these would go from light to dark.
  */

  --jp-layout-color0: white;
  --jp-layout-color1: white;
  --jp-layout-color2: var(--md-grey-200);
  --jp-layout-color3: var(--md-grey-400);

  /* Brand/accent */

  --jp-brand-color0: var(--md-blue-700);
  --jp-brand-color1: var(--md-blue-500);
  --jp-brand-color2: var(--md-blue-300);
  --jp-brand-color3: var(--md-blue-100);

  --jp-accent-color0: var(--md-green-700);
  --jp-accent-color1: var(--md-green-500);
  --jp-accent-color2: var(--md-green-300);
  --jp-accent-color3: var(--md-green-100);

  /* State colors (warn, error, success, info) */

  --jp-warn-color0: var(--md-orange-700);
  --jp-warn-color1: var(--md-orange-500);
  --jp-warn-color2: var(--md-orange-300);
  --jp-warn-color3: var(--md-orange-100);

  --jp-error-color0: var(--md-red-700);
  --jp-error-color1: var(--md-red-500);
  --jp-error-color2: var(--md-red-300);
  --jp-error-color3: var(--md-red-100);

  --jp-success-color0: var(--md-green-700);
  --jp-success-color1: var(--md-green-500);
  --jp-success-color2: var(--md-green-300);
  --jp-success-color3: var(--md-green-100);

  --jp-info-color0: var(--md-cyan-700);
  --jp-info-color1: var(--md-cyan-500);
  --jp-info-color2: var(--md-cyan-300);
  --jp-info-color3: var(--md-cyan-100);

  /* Cell specific styles */

  --jp-cell-padding: 5px;
  --jp-cell-editor-background: #f7f7f7;
  --jp-cell-editor-border-color: #cfcfcf;
  --jp-cell-editor-background-edit: var(--jp-ui-layout-color1);
  --jp-cell-editor-border-color-edit: var(--jp-brand-color1);
  --jp-cell-prompt-width: 100px;
  --jp-cell-prompt-font-family: 'Roboto Mono', monospace;
  --jp-cell-prompt-letter-spacing: 0px;
  --jp-cell-prompt-opacity: 1.0;
  --jp-cell-prompt-opacity-not-active: 0.4;
  --jp-cell-prompt-font-color-not-active: var(--md-grey-700);
  /* A custom blend of MD grey and blue 600
   * See https://meyerweb.com/eric/tools/color-blend/#546E7A:1E88E5:5:hex */
  --jp-cell-inprompt-font-color: #307FC1;
  /* A custom blend of MD grey and orange 600
   * https://meyerweb.com/eric/tools/color-blend/#546E7A:F4511E:5:hex */
  --jp-cell-outprompt-font-color: #BF5B3D;

  /* Notebook specific styles */

  --jp-notebook-padding: 10px;
  --jp-notebook-scroll-padding: 100px;

  /* Console specific styles */

  --jp-console-background: var(--md-grey-100);

  /* Toolbar specific styles */

  --jp-toolbar-border-color: var(--md-grey-400);
  --jp-toolbar-micro-height: 8px;
  --jp-toolbar-background: var(--jp-layout-color0);
  --jp-toolbar-box-shadow: 0px 0px 2px 0px rgba(0,0,0,0.24);
  --jp-toolbar-header-margin: 4px 4px 0px 4px;
  --jp-toolbar-active-background: var(--md-grey-300);
}
</style><style type="text/css">/* This file has code derived from PhosphorJS CSS files, as noted below. The license for this PhosphorJS code is:

Copyright (c) 2014-2017, PhosphorJS Contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/*
 * The following section is derived from https://github.com/phosphorjs/phosphor/blob/23b9d075ebc5b73ab148b6ebfc20af97f85714c4/packages/widgets/style/tabbar.css 
 * We've scoped the rules so that they are consistent with exactly our code.
 */

.jupyter-widgets.widget-tab > .p-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


.jupyter-widgets.widget-tab > .p-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
}


.jupyter-widgets.widget-tab > .p-TabBar[data-orientation='vertical'] {
  flex-direction: column;
}


.jupyter-widgets.widget-tab > .p-TabBar > .p-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}


.jupyter-widgets.widget-tab > .p-TabBar[data-orientation='horizontal'] > .p-TabBar-content {
  flex-direction: row;
}


.jupyter-widgets.widget-tab > .p-TabBar[data-orientation='vertical'] > .p-TabBar-content {
  flex-direction: column;
}


.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
}


.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabIcon,
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}


.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}


.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab.p-mod-hidden {
  display: none !important;
}


.jupyter-widgets.widget-tab > .p-TabBar.p-mod-dragging .p-TabBar-tab {
  position: relative;
}


.jupyter-widgets.widget-tab > .p-TabBar.p-mod-dragging[data-orientation='horizontal'] .p-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}


.jupyter-widgets.widget-tab > .p-TabBar.p-mod-dragging[data-orientation='vertical'] .p-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}


.jupyter-widgets.widget-tab > .p-TabBar.p-mod-dragging .p-TabBar-tab.p-mod-dragging {
  transition: none;
}

/* End tabbar.css */
</style><style type="text/css">/* Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*
 * We assume that the CSS variables in
 * https://github.com/jupyterlab/jupyterlab/blob/master/src/default-theme/variables.css
 * have been defined.
 */

:root {
    --jp-widgets-color: var(--jp-content-font-color1);
    --jp-widgets-label-color: var(--jp-widgets-color);
    --jp-widgets-readout-color: var(--jp-widgets-color);
    --jp-widgets-font-size: var(--jp-ui-font-size1);
    --jp-widgets-margin: 2px;
    --jp-widgets-inline-height: 28px;
    --jp-widgets-inline-width: 300px;
    --jp-widgets-inline-width-short: calc(var(--jp-widgets-inline-width) / 2 - var(--jp-widgets-margin));
    --jp-widgets-inline-width-tiny: calc(var(--jp-widgets-inline-width-short) / 2 - var(--jp-widgets-margin));
    --jp-widgets-inline-margin: 4px; /* margin between inline elements */
    --jp-widgets-inline-label-width: 80px;
    --jp-widgets-border-width: var(--jp-border-width);
    --jp-widgets-vertical-height: 200px;
    --jp-widgets-horizontal-tab-height: 24px;
    --jp-widgets-horizontal-tab-width: 144px;
    --jp-widgets-horizontal-tab-top-border: 2px;
    --jp-widgets-progress-thickness: 20px;
    --jp-widgets-container-padding: 15px;
    --jp-widgets-input-padding: 4px;
    --jp-widgets-radio-item-height-adjustment: 8px;
    --jp-widgets-radio-item-height: calc(var(--jp-widgets-inline-height) - var(--jp-widgets-radio-item-height-adjustment));
    --jp-widgets-slider-track-thickness: 4px;
    --jp-widgets-slider-border-width: var(--jp-widgets-border-width);
    --jp-widgets-slider-handle-size: 16px;
    --jp-widgets-slider-handle-border-color: var(--jp-border-color1);
    --jp-widgets-slider-handle-background-color: var(--jp-layout-color1);
    --jp-widgets-slider-active-handle-color: var(--jp-brand-color1);
    --jp-widgets-menu-item-height: 24px;
    --jp-widgets-dropdown-arrow: url("data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjIuMSwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPgo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxheWVyXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCAxOCAxOCIgc3R5bGU9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgMTggMTg7IiB4bWw6c3BhY2U9InByZXNlcnZlIj4KPHN0eWxlIHR5cGU9InRleHQvY3NzIj4KCS5zdDB7ZmlsbDpub25lO30KPC9zdHlsZT4KPHBhdGggZD0iTTUuMiw1LjlMOSw5LjdsMy44LTMuOGwxLjIsMS4ybC00LjksNWwtNC45LTVMNS4yLDUuOXoiLz4KPHBhdGggY2xhc3M9InN0MCIgZD0iTTAtMC42aDE4djE4SDBWLTAuNnoiLz4KPC9zdmc+Cg");
    --jp-widgets-input-color: var(--jp-ui-font-color1);
    --jp-widgets-input-background-color: var(--jp-layout-color1);
    --jp-widgets-input-border-color: var(--jp-border-color1);
    --jp-widgets-input-focus-border-color: var(--jp-brand-color2);
    --jp-widgets-input-border-width: var(--jp-widgets-border-width);
    --jp-widgets-disabled-opacity: 0.6;

    /* From Material Design Lite */
    --md-shadow-key-umbra-opacity: 0.2;
    --md-shadow-key-penumbra-opacity: 0.14;
    --md-shadow-ambient-shadow-opacity: 0.12;
}

.jupyter-widgets {
    margin: var(--jp-widgets-margin);
    box-sizing: border-box;
    color: var(--jp-widgets-color);
    overflow: visible;
}

.jupyter-widgets.jupyter-widgets-disconnected::before {
    line-height: var(--jp-widgets-inline-height);
    height: var(--jp-widgets-inline-height);
}

.jp-Output-result > .jupyter-widgets {
    margin-left: 0;
    margin-right: 0;
}

/* vbox and hbox */

.widget-inline-hbox {
    /* Horizontal widgets */
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: baseline;
}

.widget-inline-vbox {
    /* Vertical Widgets */
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.widget-box {
    box-sizing: border-box;
    display: flex;
    margin: 0;
    overflow: auto;
}

.widget-gridbox {
    box-sizing: border-box;
    display: grid;
    margin: 0;
    overflow: auto;
}

.widget-hbox {
    flex-direction: row;
}

.widget-vbox {
    flex-direction: column;
}

/* General Button Styling */

.jupyter-button {
    padding-left: 10px;
    padding-right: 10px;
    padding-top: 0px;
    padding-bottom: 0px;
    display: inline-block;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: center;
    font-size: var(--jp-widgets-font-size);
    cursor: pointer;

    height: var(--jp-widgets-inline-height);
    border: 0px solid;
    line-height: var(--jp-widgets-inline-height);
    box-shadow: none;

    color: var(--jp-ui-font-color1);
    background-color: var(--jp-layout-color2);
    border-color: var(--jp-border-color2);
    border: none;
    user-select: none;
}

.jupyter-button i.fa {
    margin-right: var(--jp-widgets-inline-margin);
    pointer-events: none;
}

.jupyter-button:empty:before {
    content: "\200b"; /* zero-width space */
}

.jupyter-widgets.jupyter-button:disabled {
    opacity: var(--jp-widgets-disabled-opacity);
}

.jupyter-button i.fa.center {
    margin-right: 0;
}

.jupyter-button:hover:enabled, .jupyter-button:focus:enabled {
    /* MD Lite 2dp shadow */
    box-shadow: 0 2px 2px 0 rgba(0, 0, 0, var(--md-shadow-key-penumbra-opacity)),
                0 3px 1px -2px rgba(0, 0, 0, var(--md-shadow-key-umbra-opacity)),
                0 1px 5px 0 rgba(0, 0, 0, var(--md-shadow-ambient-shadow-opacity));
}

.jupyter-button:active, .jupyter-button.mod-active {
    /* MD Lite 4dp shadow */
    box-shadow: 0 4px 5px 0 rgba(0, 0, 0, var(--md-shadow-key-penumbra-opacity)),
                0 1px 10px 0 rgba(0, 0, 0, var(--md-shadow-ambient-shadow-opacity)),
                0 2px 4px -1px rgba(0, 0, 0, var(--md-shadow-key-umbra-opacity));
    color: var(--jp-ui-font-color1);
    background-color: var(--jp-layout-color3);
}

.jupyter-button:focus:enabled {
    outline: 1px solid var(--jp-widgets-input-focus-border-color);
}

/* Button "Primary" Styling */

.jupyter-button.mod-primary {
    color: var(--jp-inverse-ui-font-color1);
    background-color: var(--jp-brand-color1);
}

.jupyter-button.mod-primary.mod-active {
    color: var(--jp-inverse-ui-font-color0);
    background-color: var(--jp-brand-color0);
}

.jupyter-button.mod-primary:active {
    color: var(--jp-inverse-ui-font-color0);
    background-color: var(--jp-brand-color0);
}

/* Button "Success" Styling */

.jupyter-button.mod-success {
    color: var(--jp-inverse-ui-font-color1);
    background-color: var(--jp-success-color1);
}

.jupyter-button.mod-success.mod-active {
    color: var(--jp-inverse-ui-font-color0);
    background-color: var(--jp-success-color0);
 }

.jupyter-button.mod-success:active {
    color: var(--jp-inverse-ui-font-color0);
    background-color: var(--jp-success-color0);
 }

 /* Button "Info" Styling */

.jupyter-button.mod-info {
    color: var(--jp-inverse-ui-font-color1);
    background-color: var(--jp-info-color1);
}

.jupyter-button.mod-info.mod-active {
    color: var(--jp-inverse-ui-font-color0);
    background-color: var(--jp-info-color0);
}

.jupyter-button.mod-info:active {
    color: var(--jp-inverse-ui-font-color0);
    background-color: var(--jp-info-color0);
}

/* Button "Warning" Styling */

.jupyter-button.mod-warning {
    color: var(--jp-inverse-ui-font-color1);
    background-color: var(--jp-warn-color1);
}

.jupyter-button.mod-warning.mod-active {
    color: var(--jp-inverse-ui-font-color0);
    background-color: var(--jp-warn-color0);
}

.jupyter-button.mod-warning:active {
    color: var(--jp-inverse-ui-font-color0);
    background-color: var(--jp-warn-color0);
}

/* Button "Danger" Styling */

.jupyter-button.mod-danger {
    color: var(--jp-inverse-ui-font-color1);
    background-color: var(--jp-error-color1);
}

.jupyter-button.mod-danger.mod-active {
    color: var(--jp-inverse-ui-font-color0);
    background-color: var(--jp-error-color0);
}

.jupyter-button.mod-danger:active {
    color: var(--jp-inverse-ui-font-color0);
    background-color: var(--jp-error-color0);
}

/* Widget Button, Widget Toggle Button, Widget Upload */

.widget-button, .widget-toggle-button, .widget-upload {
    width: var(--jp-widgets-inline-width-short);
}

/* Widget Label Styling */

/* Override Bootstrap label css */
.jupyter-widgets label {
    margin-bottom: initial;
}

.widget-label-basic {
    /* Basic Label */
    color: var(--jp-widgets-label-color);
    font-size: var(--jp-widgets-font-size);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    line-height: var(--jp-widgets-inline-height);
}

.widget-label {
    /* Label */
    color: var(--jp-widgets-label-color);
    font-size: var(--jp-widgets-font-size);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    line-height: var(--jp-widgets-inline-height);
}

.widget-inline-hbox .widget-label {
    /* Horizontal Widget Label */
    color: var(--jp-widgets-label-color);
    text-align: right;
    margin-right: calc( var(--jp-widgets-inline-margin) * 2 );
    width: var(--jp-widgets-inline-label-width);
    flex-shrink: 0;
}

.widget-inline-vbox .widget-label {
    /* Vertical Widget Label */
    color: var(--jp-widgets-label-color);
    text-align: center;
    line-height: var(--jp-widgets-inline-height);
}

/* Widget Readout Styling */

.widget-readout {
    color: var(--jp-widgets-readout-color);
    font-size: var(--jp-widgets-font-size);
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
    overflow: hidden;
    white-space: nowrap;
    text-align: center;
}

.widget-readout.overflow {
    /* Overflowing Readout */

    /* From Material Design Lite
        shadow-key-umbra-opacity: 0.2;
        shadow-key-penumbra-opacity: 0.14;
        shadow-ambient-shadow-opacity: 0.12;
     */
    -webkit-box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.2),
                        0 3px 1px -2px rgba(0, 0, 0, 0.14),
                        0 1px 5px 0 rgba(0, 0, 0, 0.12);

    -moz-box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.2),
                     0 3px 1px -2px rgba(0, 0, 0, 0.14),
                     0 1px 5px 0 rgba(0, 0, 0, 0.12);

    box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.2),
                0 3px 1px -2px rgba(0, 0, 0, 0.14),
                0 1px 5px 0 rgba(0, 0, 0, 0.12);
}

.widget-inline-hbox .widget-readout {
    /* Horizontal Readout */
    text-align: center;
    max-width: var(--jp-widgets-inline-width-short);
    min-width: var(--jp-widgets-inline-width-tiny);
    margin-left: var(--jp-widgets-inline-margin);
}

.widget-inline-vbox .widget-readout {
    /* Vertical Readout */
    margin-top: var(--jp-widgets-inline-margin);
    /* as wide as the widget */
    width: inherit;
}

/* Widget Checkbox Styling */

.widget-checkbox {
    width: var(--jp-widgets-inline-width);
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
}

.widget-checkbox input[type="checkbox"] {
    margin: 0px calc( var(--jp-widgets-inline-margin) * 2 ) 0px 0px;
    line-height: var(--jp-widgets-inline-height);
    font-size: large;
    flex-grow: 1;
    flex-shrink: 0;
    align-self: center;
}

/* Widget Valid Styling */

.widget-valid {
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
    width: var(--jp-widgets-inline-width-short);
    font-size: var(--jp-widgets-font-size);
}

.widget-valid i:before {
    line-height: var(--jp-widgets-inline-height);
    margin-right: var(--jp-widgets-inline-margin);
    margin-left: var(--jp-widgets-inline-margin);

    /* from the fa class in FontAwesome: https://github.com/FortAwesome/Font-Awesome/blob/49100c7c3a7b58d50baa71efef11af41a66b03d3/css/font-awesome.css#L14 */
    display: inline-block;
    font: normal normal normal 14px/1 FontAwesome;
    font-size: inherit;
    text-rendering: auto;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.widget-valid.mod-valid i:before {
    content: "\f00c";
    color: green;
}

.widget-valid.mod-invalid i:before {
    content: "\f00d";
    color: red;
}

.widget-valid.mod-valid .widget-valid-readout {
    display: none;
}

/* Widget Text and TextArea Stying */

.widget-textarea, .widget-text {
    width: var(--jp-widgets-inline-width);
}

.widget-text input[type="text"], .widget-text input[type="number"]{
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
}

.widget-text input[type="text"]:disabled, .widget-text input[type="number"]:disabled, .widget-textarea textarea:disabled {
    opacity: var(--jp-widgets-disabled-opacity);
}

.widget-text input[type="text"], .widget-text input[type="number"], .widget-textarea textarea {
    box-sizing: border-box;
    border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
    background-color: var(--jp-widgets-input-background-color);
    color: var(--jp-widgets-input-color);
    font-size: var(--jp-widgets-font-size);
    flex-grow: 1;
    min-width: 0; /* This makes it possible for the flexbox to shrink this input */
    flex-shrink: 1;
    outline: none !important;
}
    
.widget-text input[type="text"], .widget-textarea textarea {
    padding: var(--jp-widgets-input-padding) calc( var(--jp-widgets-input-padding) *  2);
}

.widget-text input[type="number"] {
    padding: var(--jp-widgets-input-padding) 0 var(--jp-widgets-input-padding) calc(var(--jp-widgets-input-padding) *  2);
}

.widget-textarea textarea {
    height: inherit;
    width: inherit;
}

.widget-text input:focus, .widget-textarea textarea:focus {
    border-color: var(--jp-widgets-input-focus-border-color);
}

/* Widget Slider */

.widget-slider .ui-slider {
    /* Slider Track */
    border: var(--jp-widgets-slider-border-width) solid var(--jp-layout-color3);
    background: var(--jp-layout-color3);
    box-sizing: border-box;
    position: relative;
    border-radius: 0px;
}

.widget-slider .ui-slider .ui-slider-handle {
    /* Slider Handle */
    outline: none !important; /* focused slider handles are colored - see below */
    position: absolute;
    background-color: var(--jp-widgets-slider-handle-background-color);
    border: var(--jp-widgets-slider-border-width) solid var(--jp-widgets-slider-handle-border-color);
    box-sizing: border-box;
    z-index: 1;
    background-image: none; /* Override jquery-ui */
}

/* Override jquery-ui */
.widget-slider .ui-slider .ui-slider-handle:hover, .widget-slider .ui-slider .ui-slider-handle:focus {
    background-color: var(--jp-widgets-slider-active-handle-color);
    border: var(--jp-widgets-slider-border-width) solid var(--jp-widgets-slider-active-handle-color);
}

.widget-slider .ui-slider .ui-slider-handle:active {
    background-color: var(--jp-widgets-slider-active-handle-color);
    border-color: var(--jp-widgets-slider-active-handle-color);
    z-index: 2;
    transform: scale(1.2);
}

.widget-slider  .ui-slider .ui-slider-range {
    /* Interval between the two specified value of a double slider */
    position: absolute;
    background: var(--jp-widgets-slider-active-handle-color);
    z-index: 0;
}

/* Shapes of Slider Handles */

.widget-hslider .ui-slider .ui-slider-handle {
    width: var(--jp-widgets-slider-handle-size);
    height: var(--jp-widgets-slider-handle-size);
    margin-top: calc((var(--jp-widgets-slider-track-thickness) - var(--jp-widgets-slider-handle-size)) / 2 - var(--jp-widgets-slider-border-width));
    margin-left: calc(var(--jp-widgets-slider-handle-size) / -2 + var(--jp-widgets-slider-border-width));
    border-radius: 50%;
    top: 0;
}

.widget-vslider .ui-slider .ui-slider-handle {
    width: var(--jp-widgets-slider-handle-size);
    height: var(--jp-widgets-slider-handle-size);
    margin-bottom: calc(var(--jp-widgets-slider-handle-size) / -2 + var(--jp-widgets-slider-border-width));
    margin-left: calc((var(--jp-widgets-slider-track-thickness) - var(--jp-widgets-slider-handle-size)) / 2 - var(--jp-widgets-slider-border-width));
    border-radius: 50%;
    left: 0;
}

.widget-hslider .ui-slider .ui-slider-range {
    height: calc( var(--jp-widgets-slider-track-thickness) * 2 );
    margin-top: calc((var(--jp-widgets-slider-track-thickness) - var(--jp-widgets-slider-track-thickness) * 2 ) / 2 - var(--jp-widgets-slider-border-width));
}

.widget-vslider .ui-slider .ui-slider-range {
    width: calc( var(--jp-widgets-slider-track-thickness) * 2 );
    margin-left: calc((var(--jp-widgets-slider-track-thickness) - var(--jp-widgets-slider-track-thickness) * 2 ) / 2 - var(--jp-widgets-slider-border-width));
}

/* Horizontal Slider */

.widget-hslider {
    width: var(--jp-widgets-inline-width);
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);

    /* Override the align-items baseline. This way, the description and readout
    still seem to align their baseline properly, and we don't have to have
    align-self: stretch in the .slider-container. */
    align-items: center;
}

.widgets-slider .slider-container {
    overflow: visible;
}

.widget-hslider .slider-container {
    height: var(--jp-widgets-inline-height);
    margin-left: calc(var(--jp-widgets-slider-handle-size) / 2 - 2 * var(--jp-widgets-slider-border-width));
    margin-right: calc(var(--jp-widgets-slider-handle-size) / 2 - 2 * var(--jp-widgets-slider-border-width));
    flex: 1 1 var(--jp-widgets-inline-width-short);
}

.widget-hslider .ui-slider {
    /* Inner, invisible slide div */
    height: var(--jp-widgets-slider-track-thickness);
    margin-top: calc((var(--jp-widgets-inline-height) - var(--jp-widgets-slider-track-thickness)) / 2);
    width: 100%;
}

/* Vertical Slider */

.widget-vbox .widget-label {
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
}

.widget-vslider {
    /* Vertical Slider */
    height: var(--jp-widgets-vertical-height);
    width: var(--jp-widgets-inline-width-tiny);
}

.widget-vslider .slider-container {
    flex: 1 1 var(--jp-widgets-inline-width-short);
    margin-left: auto;
    margin-right: auto;
    margin-bottom: calc(var(--jp-widgets-slider-handle-size) / 2 - 2 * var(--jp-widgets-slider-border-width));
    margin-top: calc(var(--jp-widgets-slider-handle-size) / 2 - 2 * var(--jp-widgets-slider-border-width));
    display: flex;
    flex-direction: column;
}

.widget-vslider .ui-slider-vertical {
    /* Inner, invisible slide div */
    width: var(--jp-widgets-slider-track-thickness);
    flex-grow: 1;
    margin-left: auto;
    margin-right: auto;
}

/* Widget Progress Styling */

.progress-bar {
    -webkit-transition: none;
    -moz-transition: none;
    -ms-transition: none;
    -o-transition: none;
    transition: none;
}

.progress-bar {
    height: var(--jp-widgets-inline-height);
}

.progress-bar {
    background-color: var(--jp-brand-color1);
}

.progress-bar-success {
    background-color: var(--jp-success-color1);
}

.progress-bar-info {
    background-color: var(--jp-info-color1);
}

.progress-bar-warning {
    background-color: var(--jp-warn-color1);
}

.progress-bar-danger {
    background-color: var(--jp-error-color1);
}

.progress {
    background-color: var(--jp-layout-color2);
    border: none;
    box-shadow: none;
}

/* Horisontal Progress */

.widget-hprogress {
    /* Progress Bar */
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
    width: var(--jp-widgets-inline-width);
    align-items: center;

}

.widget-hprogress .progress {
    flex-grow: 1;
    margin-top: var(--jp-widgets-input-padding);
    margin-bottom: var(--jp-widgets-input-padding);
    align-self: stretch;
    /* Override bootstrap style */
    height: initial;
}

/* Vertical Progress */

.widget-vprogress {
    height: var(--jp-widgets-vertical-height);
    width: var(--jp-widgets-inline-width-tiny);
}

.widget-vprogress .progress {
    flex-grow: 1;
    width: var(--jp-widgets-progress-thickness);
    margin-left: auto;
    margin-right: auto;
    margin-bottom: 0;
}

/* Select Widget Styling */

.widget-dropdown {
    height: var(--jp-widgets-inline-height);
    width: var(--jp-widgets-inline-width);
    line-height: var(--jp-widgets-inline-height);
}

.widget-dropdown > select {
    padding-right: 20px;
    border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
    border-radius: 0;
    height: inherit;
    flex: 1 1 var(--jp-widgets-inline-width-short);
    min-width: 0; /* This makes it possible for the flexbox to shrink this input */
    box-sizing: border-box;
    outline: none !important;
    box-shadow: none;
    background-color: var(--jp-widgets-input-background-color);
    color: var(--jp-widgets-input-color);
    font-size: var(--jp-widgets-font-size);
    vertical-align: top;
    padding-left: calc( var(--jp-widgets-input-padding) * 2);
	appearance: none;
	-webkit-appearance: none;
	-moz-appearance: none;
    background-repeat: no-repeat;
	background-size: 20px;
	background-position: right center;
    background-image: var(--jp-widgets-dropdown-arrow);
}
.widget-dropdown > select:focus {
    border-color: var(--jp-widgets-input-focus-border-color);
}

.widget-dropdown > select:disabled {
    opacity: var(--jp-widgets-disabled-opacity);
}

/* To disable the dotted border in Firefox around select controls.
   See http://stackoverflow.com/a/18853002 */
.widget-dropdown > select:-moz-focusring {
    color: transparent;
    text-shadow: 0 0 0 #000;
}

/* Select and SelectMultiple */

.widget-select {
    width: var(--jp-widgets-inline-width);
    line-height: var(--jp-widgets-inline-height);

    /* Because Firefox defines the baseline of a select as the bottom of the
    control, we align the entire control to the top and add padding to the
    select to get an approximate first line baseline alignment. */
    align-items: flex-start;
}

.widget-select > select {
    border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
    background-color: var(--jp-widgets-input-background-color);
    color: var(--jp-widgets-input-color);
    font-size: var(--jp-widgets-font-size);
    flex: 1 1 var(--jp-widgets-inline-width-short);
    outline: none !important;
    overflow: auto;
    height: inherit;

    /* Because Firefox defines the baseline of a select as the bottom of the
    control, we align the entire control to the top and add padding to the
    select to get an approximate first line baseline alignment. */
    padding-top: 5px;
}

.widget-select > select:focus {
    border-color: var(--jp-widgets-input-focus-border-color);
}

.wiget-select > select > option {
    padding-left: var(--jp-widgets-input-padding);
    line-height: var(--jp-widgets-inline-height);
    /* line-height doesn't work on some browsers for select options */
    padding-top: calc(var(--jp-widgets-inline-height)-var(--jp-widgets-font-size)/2);
    padding-bottom: calc(var(--jp-widgets-inline-height)-var(--jp-widgets-font-size)/2);
}



/* Toggle Buttons Styling */

.widget-toggle-buttons {
    line-height: var(--jp-widgets-inline-height);
}

.widget-toggle-buttons .widget-toggle-button {
    margin-left: var(--jp-widgets-margin);
    margin-right: var(--jp-widgets-margin);
}

.widget-toggle-buttons .jupyter-button:disabled {
    opacity: var(--jp-widgets-disabled-opacity);
}

/* Radio Buttons Styling */

.widget-radio {
    width: var(--jp-widgets-inline-width);
    line-height: var(--jp-widgets-inline-height);
}

.widget-radio-box {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    box-sizing: border-box;
    flex-grow: 1;
    margin-bottom: var(--jp-widgets-radio-item-height-adjustment);
}

.widget-radio-box label {
    height: var(--jp-widgets-radio-item-height);
    line-height: var(--jp-widgets-radio-item-height);
    font-size: var(--jp-widgets-font-size);
}

.widget-radio-box input {
    height: var(--jp-widgets-radio-item-height);
    line-height: var(--jp-widgets-radio-item-height);
    margin: 0 calc( var(--jp-widgets-input-padding) * 2 ) 0 1px;
    float: left;
}

/* Color Picker Styling */

.widget-colorpicker {
    width: var(--jp-widgets-inline-width);
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
}

.widget-colorpicker > .widget-colorpicker-input {
    flex-grow: 1;
    flex-shrink: 1;
    min-width: var(--jp-widgets-inline-width-tiny);
}

.widget-colorpicker input[type="color"] {
    width: var(--jp-widgets-inline-height);
    height: var(--jp-widgets-inline-height);
    padding: 0 2px; /* make the color square actually square on Chrome on OS X */
    background: var(--jp-widgets-input-background-color);
    color: var(--jp-widgets-input-color);
    border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
    border-left: none;
    flex-grow: 0;
    flex-shrink: 0;
    box-sizing: border-box;
    align-self: stretch;
    outline: none !important;
}

.widget-colorpicker.concise input[type="color"] {
    border-left: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
}

.widget-colorpicker input[type="color"]:focus, .widget-colorpicker input[type="text"]:focus {
    border-color: var(--jp-widgets-input-focus-border-color);
}

.widget-colorpicker input[type="text"] {
    flex-grow: 1;
    outline: none !important;
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
    background: var(--jp-widgets-input-background-color);
    color: var(--jp-widgets-input-color);
    border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
    font-size: var(--jp-widgets-font-size);
    padding: var(--jp-widgets-input-padding) calc( var(--jp-widgets-input-padding) *  2 );
    min-width: 0; /* This makes it possible for the flexbox to shrink this input */
    flex-shrink: 1;
    box-sizing: border-box;
}

.widget-colorpicker input[type="text"]:disabled {
    opacity: var(--jp-widgets-disabled-opacity);
}

/* Date Picker Styling */

.widget-datepicker {
    width: var(--jp-widgets-inline-width);
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
}

.widget-datepicker input[type="date"] {
    flex-grow: 1;
    flex-shrink: 1;
    min-width: 0; /* This makes it possible for the flexbox to shrink this input */
    outline: none !important;
    height: var(--jp-widgets-inline-height);
    border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
    background-color: var(--jp-widgets-input-background-color);
    color: var(--jp-widgets-input-color);
    font-size: var(--jp-widgets-font-size);
    padding: var(--jp-widgets-input-padding) calc( var(--jp-widgets-input-padding) *  2 );
    box-sizing: border-box;
}

.widget-datepicker input[type="date"]:focus {
    border-color: var(--jp-widgets-input-focus-border-color);
}

.widget-datepicker input[type="date"]:invalid {
    border-color: var(--jp-warn-color1);
}

.widget-datepicker input[type="date"]:disabled {
    opacity: var(--jp-widgets-disabled-opacity);
}

/* Play Widget */

.widget-play {
    width: var(--jp-widgets-inline-width-short);
    display: flex;
    align-items: stretch;
}

.widget-play .jupyter-button {
    flex-grow: 1;
    height: auto;
}

.widget-play .jupyter-button:disabled {
    opacity: var(--jp-widgets-disabled-opacity);
}

/* Tab Widget */

.jupyter-widgets.widget-tab {
    display: flex;
    flex-direction: column;
}

.jupyter-widgets.widget-tab > .p-TabBar {
    /* Necessary so that a tab can be shifted down to overlay the border of the box below. */
    overflow-x: visible;
    overflow-y: visible;
}

.jupyter-widgets.widget-tab > .p-TabBar > .p-TabBar-content {
    /* Make sure that the tab grows from bottom up */
    align-items: flex-end;
    min-width: 0;
    min-height: 0;
}

.jupyter-widgets.widget-tab > .widget-tab-contents {
    width: 100%;
    box-sizing: border-box;
    margin: 0;
    background: var(--jp-layout-color1);
    color: var(--jp-ui-font-color1);
    border: var(--jp-border-width) solid var(--jp-border-color1);
    padding: var(--jp-widgets-container-padding);
    flex-grow: 1;
    overflow: auto;
}

.jupyter-widgets.widget-tab > .p-TabBar {
    font: var(--jp-widgets-font-size) Helvetica, Arial, sans-serif;
    min-height: calc(var(--jp-widgets-horizontal-tab-height) + var(--jp-border-width));
}

.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab {
    flex: 0 1 var(--jp-widgets-horizontal-tab-width);
    min-width: 35px;
    min-height: calc(var(--jp-widgets-horizontal-tab-height) + var(--jp-border-width));
    line-height: var(--jp-widgets-horizontal-tab-height);
    margin-left: calc(-1 * var(--jp-border-width));
    padding: 0px 10px;
    background: var(--jp-layout-color2);
    color: var(--jp-ui-font-color2);
    border: var(--jp-border-width) solid var(--jp-border-color1);
    border-bottom: none;
    position: relative;
}

.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab.p-mod-current {
    color: var(--jp-ui-font-color0);
    /* We want the background to match the tab content background */
    background: var(--jp-layout-color1);
    min-height: calc(var(--jp-widgets-horizontal-tab-height) + 2 * var(--jp-border-width));
    transform: translateY(var(--jp-border-width));
    overflow: visible;
}

.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab.p-mod-current:before {
    position: absolute;
    top: calc(-1 * var(--jp-border-width));
    left: calc(-1 * var(--jp-border-width));
    content: '';
    height: var(--jp-widgets-horizontal-tab-top-border);
    width: calc(100% + 2 * var(--jp-border-width));
    background: var(--jp-brand-color1);
}

.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab:first-child {
    margin-left: 0;
}

.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab:hover:not(.p-mod-current) {
    background: var(--jp-layout-color1);
    color: var(--jp-ui-font-color1);
}

.jupyter-widgets.widget-tab > .p-TabBar .p-mod-closable > .p-TabBar-tabCloseIcon {
    margin-left: 4px;
}

.jupyter-widgets.widget-tab > .p-TabBar .p-mod-closable > .p-TabBar-tabCloseIcon:before {
    font-family: FontAwesome;
    content: '\f00d'; /* close */
}

.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabIcon,
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabLabel,
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabCloseIcon {
    line-height: var(--jp-widgets-horizontal-tab-height);
}

/* Accordion Widget */

.p-Collapse {
    display: flex;
    flex-direction: column;
    align-items: stretch;
}

.p-Collapse-header {
    padding: var(--jp-widgets-input-padding);
    cursor: pointer;
    color: var(--jp-ui-font-color2);
    background-color: var(--jp-layout-color2);
    border: var(--jp-widgets-border-width) solid var(--jp-border-color1);
    padding: calc(var(--jp-widgets-container-padding) * 2 / 3) var(--jp-widgets-container-padding);
    font-weight: bold;
}

.p-Collapse-header:hover {
    background-color: var(--jp-layout-color1);
    color: var(--jp-ui-font-color1);
}

.p-Collapse-open > .p-Collapse-header {
    background-color: var(--jp-layout-color1);
    color: var(--jp-ui-font-color0);
    cursor: default;
    border-bottom: none;
}

.p-Collapse .p-Collapse-header::before {
    content: '\f0da\00A0';  /* caret-right, non-breaking space */
    display: inline-block;
    font: normal normal normal 14px/1 FontAwesome;
    font-size: inherit;
    text-rendering: auto;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.p-Collapse-open > .p-Collapse-header::before {
    content: '\f0d7\00A0'; /* caret-down, non-breaking space */
}

.p-Collapse-contents {
    padding: var(--jp-widgets-container-padding);
    background-color: var(--jp-layout-color1);
    color: var(--jp-ui-font-color1);
    border-left: var(--jp-widgets-border-width) solid var(--jp-border-color1);
    border-right: var(--jp-widgets-border-width) solid var(--jp-border-color1);
    border-bottom: var(--jp-widgets-border-width) solid var(--jp-border-color1);
    overflow: auto;
}

.p-Accordion {
    display: flex;
    flex-direction: column;
    align-items: stretch;
}

.p-Accordion .p-Collapse {
    margin-bottom: 0;
}

.p-Accordion .p-Collapse + .p-Collapse {
    margin-top: 4px;
}



/* HTML widget */

.widget-html, .widget-htmlmath {
    font-size: var(--jp-widgets-font-size);
}

.widget-html > .widget-html-content, .widget-htmlmath > .widget-html-content {
    /* Fill out the area in the HTML widget */
    align-self: stretch;
    flex-grow: 1;
    flex-shrink: 1;
    /* Makes sure the baseline is still aligned with other elements */
    line-height: var(--jp-widgets-inline-height);
    /* Make it possible to have absolutely-positioned elements in the html */
    position: relative;
}


/* Image widget  */

.widget-image {
    max-width: 100%;
    height: auto;
}
</style><style type="text/css">/* Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */
</style><link id="favicon" type="image/x-icon" rel="shortcut icon" href="http://localhost:8888/static/base/images/favicon-notebook.ico"></head>

<body class="notebook_app edit_mode" data-jupyter-api-token="54be5d1e94bc5ca32fa82ebc8c1ebf260a3244b9fd5d6679" data-base-url="/" data-ws-url="" data-notebook-name="z.ipynb" data-notebook-path="z/z.ipynb" dir="ltr" data-gr-c-s-loaded="true"><div id="MathJax_Message" style="display: none;"></div><div style="visibility: hidden; overflow: hidden; position: absolute; top: 0px; height: 1px; width: auto; padding: 0px; border: 0px none; margin: 0px; text-align: left; text-indent: 0px; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal;"><div id="MathJax_Hidden"></div></div>

<noscript>
    <div id='noscript'>
      Jupyter Notebook requires JavaScript.<br>
      Please enable it to proceed. 
  </div>
</noscript>

<div id="header" role="navigation" aria-label="Top Menu" style="display: block;">
  <div id="header-container" class="container">
  <div id="ipython_notebook" class="nav navbar-brand"><a href="http://localhost:8888/tree?token=54be5d1e94bc5ca32fa82ebc8c1ebf260a3244b9fd5d6679" title="dashboard">
      <img src="test%20-%20Jupyter%20Notebook_files/logo.png" alt="Jupyter Notebook">
  </a></div>

  


<span id="save_widget" class="save_widget">
    <span id="notebook_name" class="filename">z</span>
    <span class="checkpoint_status" title="Mon, May 4, 2020 2:58 AM">Last Checkpoint: 12 hours ago</span>
    <span class="autosave_status">(autosaved)</span>
</span>


  

<span id="kernel_logo_widget">
  
  <img class="current_kernel_logo" alt="Current Kernel Logo" src="test%20-%20Jupyter%20Notebook_files/logo-64x64.png" title="Python 3" style="display: inline;">
  
</span>


  
  
  
  

    <span id="login_widget">
      
        <button id="logout" class="btn btn-sm navbar-btn">Logout</button>
      
    </span>

  

  
  
  </div>
  <div class="header-bar"></div>

  
<div id="menubar-container" class="container">
<div id="menubar">
    <div id="menus" class="navbar navbar-default" role="navigation">
        <div class="container-fluid">
            <button type="button" class="btn btn-default navbar-btn navbar-toggle" data-toggle="collapse" data-target=".navbar-collapse">
              <i class="fa fa-bars"></i>
              <span class="navbar-text">Menu</span>
            </button>
            <p id="kernel_indicator" class="navbar-text indicator_area">
              <span class="kernel_indicator_name">Python 3</span>
              <i id="kernel_indicator_icon" class="kernel_idle_icon" title="Kernel Idle"></i>
            </p>
            <i id="readonly-indicator" class="navbar-text" title="This notebook is read-only" style="display: none;">
                <span class="fa-stack">
                    <i class="fa fa-save fa-stack-1x"></i>
                    <i class="fa fa-ban fa-stack-2x text-danger"></i>
                </span>
            </i>
            <i id="modal_indicator" class="navbar-text modal_indicator" title="Edit Mode"></i>
            <span id="notification_area"><div id="notification_kernel" class="notification_widget btn btn-xs navbar-btn undefined info" style="display: none;"><span></span></div><div id="notification_notebook" class="notification_widget btn btn-xs navbar-btn" style="display: none;"><span></span></div><div id="notification_trusted" class="notification_widget btn btn-xs navbar-btn" style="cursor: help;" disabled="disabled"><span title="Javascript enabled for notebook display">Trusted</span></div><div id="notification_widgets" class="notification_widget btn btn-xs navbar-btn" style="display: none;"><span></span></div></span>
            <div class="navbar-collapse collapse">
              <ul class="nav navbar-nav">
                <li class="dropdown"><a href="#" id="filelink" aria-haspopup="true" aria-controls="file_menu class=" dropdown-toggle"="" data-toggle="dropdown">File</a>
                    <ul id="file_menu" class="dropdown-menu" role="menu" aria-labelledby="filelink">
                        <li id="new_notebook" class="dropdown-submenu" role="none">
                            <a href="#" role="menuitem">New Notebook<span class="sr-only">Toggle Dropdown</span></a>
                            <ul class="dropdown-menu" id="menu-new-notebook-submenu"><li id="new-notebook-submenu-python3"><a href="#">Python 3</a></li></ul>
                        </li>
                        <li id="open_notebook" role="none" title="Opens a new window with the Dashboard view">
                            <a href="#" role="menuitem">Open...</a></li>
                        <!-- <hr/> -->
                        <li class="divider" role="none"></li>
                        <li id="copy_notebook" role="none" title="Open a copy of this notebook's contents and start a new kernel">
                            <a href="#" role="menuitem">Make a Copy...</a></li>
                        <li id="save_notebook_as" role="none" title="Save a copy of the notebook's contents and start a new kernel">
                            <a href="#" role="menuitem">Save as...</a></li>
                        <li id="rename_notebook" role="none"><a href="#" role="menuitem">Rename...</a></li>
                        <li id="save_checkpoint" role="none"><a href="#" role="menuitem">Save and Checkpoint</a></li>
                        <!-- <hr/> -->
                        <li class="divider" role="none"></li>
                        <li id="restore_checkpoint" class="dropdown-submenu" role="none"><a href="#" role="menuitem">Revert to Checkpoint<span class="sr-only">Toggle Dropdown</span></a>
                          <ul class="dropdown-menu"><li><a href="#">Monday, May 4, 2020 2:58 AM</a></li></ul>
                        </li>
                        <li class="divider" role="none"></li>
                        <li id="print_preview" role="none"><a href="#" role="menuitem">Print Preview</a></li>
                        <li class="dropdown-submenu" role="none"><a href="#" role="menuitem">Download as<span class="sr-only">Toggle Dropdown</span></a>
                            <ul id="download_menu" class="dropdown-menu">
                                
                                <li id="download_asciidoc">
                                    <a href="#">AsciiDoc (.asciidoc)</a>
                                </li>
                                
                                <li id="download_html">
                                    <a href="#">HTML (.html)</a>
                                </li>
                                
                                <li id="download_latex">
                                    <a href="#">LaTeX (.tex)</a>
                                </li>
                                
                                <li id="download_markdown">
                                    <a href="#">Markdown (.md)</a>
                                </li>
                                
                                <li id="download_notebook">
                                    <a href="#">Notebook (.ipynb)</a>
                                </li>
                                
                                <li id="download_pdf">
                                    <a href="#">PDF via LaTeX (.pdf)</a>
                                </li>
                                
                                <li id="download_rst">
                                    <a href="#">reST (.rst)</a>
                                </li>
                                
                                <li id="download_script">
                                    <a href="#">Python (.py)</a>
                                </li>
                                
                                <li id="download_slides">
                                    <a href="#">Reveal.js slides (.slides.html)</a>
                                </li>
                                
                            </ul>
                        </li>
                        <li class="dropdown-submenu hidden" role="none"><a href="#" role="menuitem">Deploy as</a>
                            <ul id="deploy_menu" class="dropdown-menu"></ul>
                        </li>
                        <li class="divider" role="none"></li>
                        <li id="trust_notebook" role="none" title="Trust the output of this notebook" class="disabled">
                            <a href="#" role="menuitem">Trusted Notebook</a></li>
                        <li class="divider" role="none"></li>
                        <li id="close_and_halt" role="none" title="Shutdown this notebook's kernel, and close this window">
                            <a href="#" role="menuitem">Close and Halt</a></li>
                    </ul>
                </li>

                <li class="dropdown"><a href="#" class="dropdown-toggle" id="editlink" data-toggle="dropdown" aria-haspopup="true" aria-controls="edit_menu">Edit</a>
                    <ul id="edit_menu" class="dropdown-menu" role="menu" aria-labelledby="editlink">
                        <li id="cut_cell" role="none"><a href="#" role="menuitem">Cut Cells</a></li>
                        <li id="copy_cell" role="none"><a href="#" role="menuitem">Copy Cells</a></li>
                        <li id="paste_cell_above" class="disabled" role="none"><a href="#" role="menuitem" aria-disabled="true">Paste Cells Above</a></li>
                        <li id="paste_cell_below" class="disabled" role="none"><a href="#" role="menuitem" aria-disabled="true">Paste Cells Below</a></li>
                        <li id="paste_cell_replace" class="disabled" role="none"><a href="#" role="menuitem" aria-disabled="true">Paste Cells &amp; Replace</a></li>
                        <li id="delete_cell" role="none"><a href="#" role="menuitem">Delete Cells</a></li>
                        <li id="undelete_cell" class="disabled" role="none"><a href="#" role="menuitem" aria-disabled="true">Undo Delete Cells</a></li>
                        <li class="divider" role="none"></li>
                        <li id="split_cell" role="none"><a href="#" role="menuitem">Split Cell</a></li>
                        <li id="merge_cell_above" role="none"><a href="#" role="menuitem">Merge Cell Above</a></li>
                        <li id="merge_cell_below" role="none"><a href="#" role="menuitem">Merge Cell Below</a></li>
                        <li class="divider" role="none"></li>
                        <li id="move_cell_up" role="none"><a href="#" role="menuitem">Move Cell Up</a></li>
                        <li id="move_cell_down" role="none"><a href="#" role="menuitem">Move Cell Down</a></li>
                        <li class="divider" role="none"></li>
                        <li id="edit_nb_metadata" role="none"><a href="#" role="menuitem">Edit Notebook Metadata</a></li>
                        <li class="divider" role="none"></li>
                        <li id="find_and_replace" role="none"><a href="#" role="menuitem"> Find and Replace </a></li>
                        <li class="divider" role="none"></li>
                        <li id="cut_cell_attachments" role="none"><a href="#" role="menuitem">Cut Cell Attachments</a></li>
                        <li id="copy_cell_attachments" role="none"><a href="#" role="menuitem">Copy Cell Attachments</a></li>
                        <li id="paste_cell_attachments" class="disabled" role="none"><a href="#" role="menuitem" aria-disabled="true">Paste Cell Attachments</a></li>
                        <li class="divider" role="none"></li>
                        <li id="insert_image" class="disabled" role="none"><a href="#" role="menuitem" aria-disabled="true">  Insert Image </a></li>
                    </ul>
                </li>
                <li class="dropdown"><a href="#" class="dropdown-toggle" id="viewlink" data-toggle="dropdown" aria-haspopup="true" aria-controls="view_menu">View</a>
                    <ul id="view_menu" class="dropdown-menu" role="menu" aria-labelledby="viewlink">
                        <li id="toggle_header" role="none" title="Show/Hide the logo and notebook title (above menu bar)">
                            <a href="#" role="menuitem">Toggle Header</a>
                        </li>
                        <li id="toggle_toolbar" role="none" title="Show/Hide the action icons (below menu bar)">
                            <a href="#" role="menuitem">Toggle Toolbar</a>
                        </li>
                        <li id="toggle_line_numbers" role="none" title="Show/Hide line numbers in cells">
                            <a href="#" role="menuitem">Toggle Line Numbers</a>
                        </li>
                        <li id="menu-cell-toolbar" class="dropdown-submenu" role="none">
                            <a href="#" role="menuitem">Cell Toolbar</a>
                            <ul class="dropdown-menu" id="menu-cell-toolbar-submenu"><li data-name="None"><a href="#">None</a></li><li data-name="Edit%20Metadata"><a href="#">Edit Metadata</a></li><li data-name="Raw%20Cell%20Format"><a href="#">Raw Cell Format</a></li><li data-name="Slideshow"><a href="#">Slideshow</a></li><li data-name="Attachments"><a href="#">Attachments</a></li><li data-name="Tags"><a href="#">Tags</a></li></ul>
                        </li>
                    </ul>
                </li>
                <li class="dropdown"><a href="#" class="dropdown-toggle" id="insertlink" data-toggle="dropdown" aria-haspopup="true" aria-controls="insert_menu">Insert</a>
                    <ul id="insert_menu" class="dropdown-menu" role="menu" aria-labelledby="insertlink">
                        <li id="insert_cell_above" role="none" title="Insert an empty Code cell above the currently active cell">
                            <a href="#" role="menuitem">Insert Cell Above</a></li>
                        <li id="insert_cell_below" role="none" title="Insert an empty Code cell below the currently active cell">
                            <a href="#" role="menuitem">Insert Cell Below</a></li>
                    </ul>
                </li>
                <li class="dropdown"><a href="#" class="dropdown-toggle" data-toggle="dropdown">Cell</a>
                    <ul id="cell_menu" class="dropdown-menu">
                        <li id="run_cell" title="Run this cell, and move cursor to the next one">
                            <a href="#">Run Cells</a></li>
                        <li id="run_cell_select_below" title="Run this cell, select below">
                            <a href="#">Run Cells and Select Below</a></li>
                        <li id="run_cell_insert_below" title="Run this cell, insert below">
                            <a href="#">Run Cells and Insert Below</a></li>
                        <li id="run_all_cells" title="Run all cells in the notebook">
                            <a href="#">Run All</a></li>
                        <li id="run_all_cells_above" title="Run all cells above (but not including) this cell">
                            <a href="#">Run All Above</a></li>
                        <li id="run_all_cells_below" title="Run this cell and all cells below it">
                            <a href="#">Run All Below</a></li>
                        <li class="divider"></li>
                        <li id="change_cell_type" class="dropdown-submenu" title="All cells in the notebook have a cell type. By default, new cells are created as 'Code' cells">
                            <a href="#">Cell Type</a>
                            <ul class="dropdown-menu">
                              <li id="to_code" title="Contents will be sent to the kernel for execution, and output will display in the footer of cell">
                                  <a href="#">Code</a></li>
                              <li id="to_markdown" title="Contents will be rendered as HTML and serve as explanatory text">
                                  <a href="#">Markdown</a></li>
                              <li id="to_raw" title="Contents will pass through nbconvert unmodified">
                                  <a href="#">Raw NBConvert</a></li>
                            </ul>
                        </li>
                        <li class="divider"></li>
                        <li id="current_outputs" class="dropdown-submenu"><a href="#">Current Outputs</a>
                            <ul class="dropdown-menu">
                                <li id="toggle_current_output" title="Hide/Show the output of the current cell">
                                    <a href="#">Toggle</a>
                                </li>
                                <li id="toggle_current_output_scroll" title="Scroll the output of the current cell">
                                    <a href="#">Toggle Scrolling</a>
                                </li>
                                <li id="clear_current_output" title="Clear the output of the current cell">
                                    <a href="#">Clear</a>
                                </li>
                            </ul>
                        </li>
                        <li id="all_outputs" class="dropdown-submenu"><a href="#">All Output</a>
                            <ul class="dropdown-menu">
                                <li id="toggle_all_output" title="Hide/Show the output of all cells">
                                    <a href="#">Toggle</a>
                                </li>
                                <li id="toggle_all_output_scroll" title="Scroll the output of all cells">
                                    <a href="#">Toggle Scrolling</a>
                                </li>
                                <li id="clear_all_output" title="Clear the output of all cells">
                                    <a href="#">Clear</a>
                                </li>
                            </ul>
                        </li>
                    </ul>
                </li>
                <li class="dropdown"><a href="#" class="dropdown-toggle" data-toggle="dropdown" aria-expanded="false">Kernel</a>
                    <ul id="kernel_menu" class="dropdown-menu">
                        <li id="int_kernel" title="Send Keyboard Interrupt (CTRL-C) to the Kernel">
                            <a href="#">Interrupt</a>
                        </li>
                        <li id="restart_kernel" title="Restart the Kernel">
                            <a href="#">Restart</a>
                        </li>
                        <li id="restart_clear_output" title="Restart the Kernel and clear all output">
                            <a href="#">Restart &amp; Clear Output</a>
                        </li>
                        <li id="restart_run_all" title="Restart the Kernel and re-run the notebook">
                            <a href="#">Restart &amp; Run All</a>
                        </li>
                        <li id="reconnect_kernel" title="Reconnect to the Kernel">
                            <a href="#">Reconnect</a>
                        </li>
                        <li id="shutdown_kernel" title="Shutdown the Kernel">
                            <a href="#">Shutdown</a>
                        </li>
                        <li class="divider"></li>
                        <li id="menu-change-kernel" class="dropdown-submenu">
                            <a href="#">Change kernel</a>
                            <ul class="dropdown-menu" id="menu-change-kernel-submenu"><li id="kernel-submenu-python3"><a href="#">Python 3</a></li></ul>
                        </li>
                    </ul>
                </li>
                <li class="dropdown"><a href="#" data-toggle="dropdown" class="dropdown-toggle">Widgets</a><ul id="widget-submenu" class="dropdown-menu"><li title="Save the notebook with the widget state information for static rendering"><a href="#">Save Notebook Widget State</a></li><li title="Clear the widget state information from the notebook"><a href="#">Clear Notebook Widget State</a></li><ul class="divider"></ul><li title="Download the widget state as a JSON file"><a href="#">Download Widget State</a></li><li title="Embed interactive widgets"><a href="#">Embed Widgets</a></li></ul></li><li class="dropdown"><a href="#" class="dropdown-toggle" data-toggle="dropdown">Help</a>
                    <ul id="help_menu" class="dropdown-menu">
                        
                        <li id="notebook_tour" title="A quick tour of the notebook user interface"><a href="#">User Interface Tour</a></li>
                        <li id="keyboard_shortcuts" title="Opens a tooltip with all keyboard shortcuts"><a href="#">Keyboard Shortcuts</a></li>
                        <li id="edit_keyboard_shortcuts" title="Opens a dialog allowing you to edit Keyboard shortcuts"><a href="#">Edit Keyboard Shortcuts</a></li>
                        <li class="divider"></li>
                        

						
                        
                            
                                <li><a rel="noreferrer" href="http://nbviewer.jupyter.org/github/ipython/ipython/blob/3.x/examples/Notebook/Index.ipynb" target="_blank" title="Opens in a new window">
                                
                                    <i class="fa fa-external-link menu-icon pull-right"></i>
                                

                                Notebook Help
                                </a></li>
                            
                                <li><a rel="noreferrer" href="https://help.github.com/articles/markdown-basics/" target="_blank" title="Opens in a new window">
                                
                                    <i class="fa fa-external-link menu-icon pull-right"></i>
                                

                                Markdown
                                </a></li>
                            
                            
                        
                        <li id="kernel-help-links" class="divider"></li><li><a target="_blank" title="Opens in a new window" href="https://docs.python.org/3.7?v=20200504125913"><i class="fa fa-external-link menu-icon pull-right"></i><span>Python Reference</span></a></li><li><a target="_blank" title="Opens in a new window" href="https://ipython.org/documentation.html?v=20200504125913"><i class="fa fa-external-link menu-icon pull-right"></i><span>IPython Reference</span></a></li><li><a target="_blank" title="Opens in a new window" href="https://docs.scipy.org/doc/numpy/reference/?v=20200504125913"><i class="fa fa-external-link menu-icon pull-right"></i><span>NumPy Reference</span></a></li><li><a target="_blank" title="Opens in a new window" href="https://docs.scipy.org/doc/scipy/reference/?v=20200504125913"><i class="fa fa-external-link menu-icon pull-right"></i><span>SciPy Reference</span></a></li><li><a target="_blank" title="Opens in a new window" href="https://matplotlib.org/contents.html?v=20200504125913"><i class="fa fa-external-link menu-icon pull-right"></i><span>Matplotlib Reference</span></a></li><li><a target="_blank" title="Opens in a new window" href="http://docs.sympy.org/latest/index.html?v=20200504125913"><i class="fa fa-external-link menu-icon pull-right"></i><span>SymPy Reference</span></a></li><li><a target="_blank" title="Opens in a new window" href="https://pandas.pydata.org/pandas-docs/stable/?v=20200504125913"><i class="fa fa-external-link menu-icon pull-right"></i><span>pandas Reference</span></a></li><li class="divider"></li>
                        <li title="About Jupyter Notebook"><a id="notebook_about" href="#">About</a></li>
                        
                    </ul>
                </li>
              </ul>
            </div>
        </div>
    </div>
</div>

<div id="maintoolbar" class="navbar">
  <div class="toolbar-inner navbar-inner navbar-nobg">
    <div id="maintoolbar-container" class="container toolbar"><div class="btn-group" id="save-notbook"><button class="btn btn-default" title="Save and Checkpoint" data-jupyter-action="jupyter-notebook:save-notebook"><i class="fa-save fa"></i></button></div><div class="btn-group" id="insert_above_below"><button class="btn btn-default" title="insert cell below" data-jupyter-action="jupyter-notebook:insert-cell-below"><i class="fa-plus fa"></i></button></div><div class="btn-group" id="cut_copy_paste"><button class="btn btn-default" title="cut selected cells" data-jupyter-action="jupyter-notebook:cut-cell"><i class="fa-cut fa"></i></button><button class="btn btn-default" title="copy selected cells" data-jupyter-action="jupyter-notebook:copy-cell"><i class="fa-copy fa"></i></button><button class="btn btn-default" title="paste cells below" data-jupyter-action="jupyter-notebook:paste-cell-below"><i class="fa-paste fa"></i></button></div><div class="btn-group" id="move_up_down"><button class="btn btn-default" title="move selected cells up" data-jupyter-action="jupyter-notebook:move-cell-up"><i class="fa-arrow-up fa"></i></button><button class="btn btn-default" title="move selected cells down" data-jupyter-action="jupyter-notebook:move-cell-down"><i class="fa-arrow-down fa"></i></button></div><div class="btn-group" id="run_int"><button class="btn btn-default" title="Run" data-jupyter-action="jupyter-notebook:run-cell-and-select-next"><i class="fa-step-forward fa"></i><span class="toolbar-btn-label">Run</span></button><button class="btn btn-default" title="interrupt the kernel" data-jupyter-action="jupyter-notebook:interrupt-kernel"><i class="fa-stop fa"></i></button><button class="btn btn-default" title="restart the kernel (with dialog)" data-jupyter-action="jupyter-notebook:confirm-restart-kernel"><i class="fa-repeat fa"></i></button><button class="btn btn-default" title="restart the kernel, then re-run the whole notebook (with dialog)" data-jupyter-action="jupyter-notebook:confirm-restart-kernel-and-run-all-cells"><i class="fa-forward fa"></i></button></div><select id="cell_type" aria-label="combobox, select cell type" role="combobox" class="form-control select-xs"><option value="code" selected="selected">Code</option><option value="markdown">Markdown</option><option value="raw">Raw NBConvert</option><option value="heading">Heading</option><option value="multiselect" disabled="disabled" style="display: none;">-</option></select><div class="btn-group" id="cmd_palette"><button class="btn btn-default" title="open the command palette" data-jupyter-action="jupyter-notebook:show-command-palette"><i class="fa-keyboard-o fa"></i></button></div></div>
  </div>
</div>
</div>

<div class="lower-header-bar"></div>

</div>

<div id="site" style="display: block; height: 668.117px;">


<div id="ipython-main-app">
    <div id="notebook_panel">
        <div id="notebook" tabindex="-1"><div class="container" id="notebook-container"><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[1]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 141.6px; left: 249.2px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true" style="bottom: 0px;"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 831.8px; margin-bottom: -16px; border-right-width: 14px; min-height: 368px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style=""><div class="CodeMirror-cursor" style="left: 249.2px; top: 136px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation" style=""><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-comment"># import the necessary packages</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">preprocessing</span> <span class="cm-keyword">import</span> <span class="cm-variable">LabelBinarizer</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">model_selection</span> <span class="cm-keyword">import</span> <span class="cm-variable">train_test_split</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">import</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">model_selection</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">metrics</span> <span class="cm-keyword">import</span> <span class="cm-variable">classification_report</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">from</span> <span class="cm-variable">keras</span>.<span class="cm-property">models</span> <span class="cm-keyword">import</span> <span class="cm-variable">Sequential</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">from</span> <span class="cm-variable">keras</span>.<span class="cm-property">models</span> <span class="cm-keyword">import</span> <span class="cm-variable">Model</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">from</span> <span class="cm-variable">keras</span>.<span class="cm-property">callbacks</span> <span class="cm-keyword">import</span> <span class="cm-variable">EarlyStopping</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">from</span> <span class="cm-variable">keras</span>.<span class="cm-property">layers</span> <span class="cm-keyword">import</span> <span class="cm-variable">Conv2D</span>, <span class="cm-variable">MaxPooling2D</span>, <span class="cm-variable">Flatten</span>, <span class="cm-variable">Dropout</span>, <span class="cm-variable">BatchNormalization</span>, <span class="cm-variable">ZeroPadding2D</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">from</span> <span class="cm-variable">keras</span>.<span class="cm-property">layers</span>.<span class="cm-property">core</span> <span class="cm-keyword">import</span> <span class="cm-variable">Dense</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">from</span> <span class="cm-variable">keras</span>.<span class="cm-property">optimizers</span> <span class="cm-keyword">import</span> <span class="cm-variable">Adam</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">from</span> <span class="cm-variable">imutils</span> <span class="cm-keyword">import</span> <span class="cm-variable">paths</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">import</span> <span class="cm-variable">matplotlib</span>.<span class="cm-property">pyplot</span> <span class="cm-keyword">as</span> <span class="cm-variable">plt</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">from</span> <span class="cm-variable">shutil</span> <span class="cm-keyword">import</span> <span class="cm-variable">copyfile</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">import</span> <span class="cm-variable">datetime</span> <span class="cm-keyword">as</span> <span class="cm-variable">dt</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">import</span> <span class="cm-variable">random</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">import</span> <span class="cm-variable">pickle</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">import</span> <span class="cm-variable">cv2</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">import</span> <span class="cm-variable">os</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">import</span> <span class="cm-variable">matplotlib</span></span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 368px;"></div><div class="CodeMirror-gutters" style="display: none; height: 382px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stderr"><pre>Using TensorFlow backend.
</pre></div></div></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[2]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 107.6px; left: 442.4px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true" style="bottom: 0px;"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 655.4px; margin-bottom: -16px; border-right-width: 14px; min-height: 504px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style=""><div class="CodeMirror-cursor" style="left: 442.4px; top: 102px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation" style=""><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">def</span> <span class="cm-def">build_model</span>(<span class="cm-variable">classes_size</span><span class="cm-operator">=</span><span class="cm-number">2</span>):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-comment"># initializing</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span> <span class="cm-operator">=</span> <span class="cm-variable">Sequential</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">Conv2D</span>(<span class="cm-number">16</span>, (<span class="cm-number">4</span>, <span class="cm-number">4</span>), <span class="cm-variable">input_shape</span><span class="cm-operator">=</span>(<span class="cm-number">64</span>, <span class="cm-number">64</span>, <span class="cm-number">3</span>), <span class="cm-variable">activation</span><span class="cm-operator">=</span><span class="cm-string">'relu'</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">ZeroPadding2D</span>(<span class="cm-variable">padding</span><span class="cm-operator">=</span>(<span class="cm-number">1</span>, <span class="cm-number">1</span>)))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span><span class=" CodeMirror-matchingbracket">(</span><span class="cm-variable">Conv2D</span>(<span class="cm-number">16</span>, (<span class="cm-number">4</span>, <span class="cm-number">4</span>), <span class="cm-variable">activation</span><span class="cm-operator">=</span><span class="cm-string">'relu'</span>)<span class=" CodeMirror-matchingbracket">)</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">MaxPooling2D</span>(<span class="cm-variable">pool_size</span><span class="cm-operator">=</span>(<span class="cm-number">2</span>, <span class="cm-number">2</span>)))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">Dropout</span>(<span class="cm-number">0.2</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">ZeroPadding2D</span>(<span class="cm-variable">padding</span><span class="cm-operator">=</span>(<span class="cm-number">1</span>, <span class="cm-number">1</span>)))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">Conv2D</span>(<span class="cm-number">32</span>, (<span class="cm-number">3</span>, <span class="cm-number">3</span>), <span class="cm-variable">activation</span><span class="cm-operator">=</span><span class="cm-string">'relu'</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">Conv2D</span>(<span class="cm-number">32</span>, (<span class="cm-number">3</span>, <span class="cm-number">3</span>), <span class="cm-variable">activation</span><span class="cm-operator">=</span><span class="cm-string">'relu'</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">MaxPooling2D</span>(<span class="cm-variable">pool_size</span><span class="cm-operator">=</span>(<span class="cm-number">2</span>, <span class="cm-number">2</span>)))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">Dropout</span>(<span class="cm-number">0.3</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">ZeroPadding2D</span>(<span class="cm-variable">padding</span><span class="cm-operator">=</span>(<span class="cm-number">1</span>, <span class="cm-number">1</span>)))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">Conv2D</span>(<span class="cm-number">64</span>, (<span class="cm-number">3</span>, <span class="cm-number">3</span>), <span class="cm-variable">activation</span><span class="cm-operator">=</span><span class="cm-string">'relu'</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">Conv2D</span>(<span class="cm-number">128</span>, (<span class="cm-number">3</span>, <span class="cm-number">3</span>), <span class="cm-variable">activation</span><span class="cm-operator">=</span><span class="cm-string">'relu'</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">MaxPooling2D</span>(<span class="cm-variable">pool_size</span><span class="cm-operator">=</span>(<span class="cm-number">2</span>, <span class="cm-number">2</span>)))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">Dropout</span>(<span class="cm-number">0.4</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">Flatten</span>())</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">Dense</span>(<span class="cm-variable">units</span><span class="cm-operator">=</span><span class="cm-number">512</span>, <span class="cm-variable">activation</span><span class="cm-operator">=</span><span class="cm-string">"relu"</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">Dropout</span>(<span class="cm-number">0.5</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">model</span>.<span class="cm-property">add</span>(<span class="cm-variable">Dense</span>(<span class="cm-variable">classes_size</span>, <span class="cm-variable">activation</span><span class="cm-operator">=</span><span class="cm-string">"softmax"</span>, <span class="cm-variable">name</span><span class="cm-operator">=</span><span class="cm-string">"out"</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-keyword">return</span> <span class="cm-variable">model</span></span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 504px;"></div><div class="CodeMirror-gutters" style="display: none; height: 518px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[3]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.60004px; left: 5.60001px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true" style="bottom: 0px;"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 689px; margin-bottom: -16px; border-right-width: 14px; min-height: 266px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors"><div class="CodeMirror-cursor" style="left: 5.60001px; top: 0px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation" style=""><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-comment"># main</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">previous_trainX</span>, <span class="cm-variable">previous_trainY</span>, <span class="cm-variable">previous_testX</span>, <span class="cm-variable">previous_testY</span> <span class="cm-operator">=</span> [], [], [], []</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">data_list</span>, <span class="cm-variable">labels_list</span> <span class="cm-operator">=</span> [], []</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">loss</span>, <span class="cm-variable">val_loss</span>, <span class="cm-variable">accuracy</span>, <span class="cm-variable">val_accuracy</span> <span class="cm-operator">=</span> [], [], [], []</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">dataSet_path</span> <span class="cm-operator">=</span> <span class="cm-string">'PlantVillage'</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">EPOCHS</span> <span class="cm-operator">=</span> <span class="cm-number">5</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">batch_size</span> <span class="cm-operator">=</span> <span class="cm-number">32</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">opt</span> <span class="cm-operator">=</span> <span class="cm-variable">Adam</span>(<span class="cm-variable">lr</span><span class="cm-operator">=</span><span class="cm-number">0.001</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">model</span> <span class="cm-operator">=</span> <span class="cm-variable">Model</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">lb</span> <span class="cm-operator">=</span> <span class="cm-variable">LabelBinarizer</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">folders</span> <span class="cm-operator">=</span> <span class="cm-variable">os</span>.<span class="cm-property">listdir</span>(<span class="cm-variable">dataSet_path</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-builtin">print</span>(<span class="cm-variable">folders</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">folders</span>[<span class="cm-number">0</span>:<span class="cm-number">2</span>] <span class="cm-operator">=</span> []</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">num_labels</span> <span class="cm-operator">=</span> <span class="cm-builtin">len</span>(<span class="cm-variable">folders</span>)</span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 266px;"></div><div class="CodeMirror-gutters" style="display: none; height: 280px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout"><pre>['Pepper_Bell_Bacterial_spot', 'Pepper_Bell_healthy', 'Potato_Early_blight', 'Potato_Healthy', 'Potato_Late_blight', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Tomato_Mosaic_virus', 'Tomato_Tomato_YellowLeaf_Curl_Virus']
</pre></div></div></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[4]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 362.6px; left: 5.60001px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true" style="bottom: 16px;"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true" style="right: 0px; left: 0px;"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true" style="height: 16px; width: 16px;"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 999.8px; margin-bottom: -16px; border-right-width: 14px; min-height: 657px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style=""><div class="CodeMirror-cursor" style="left: 5.60001px; top: 357px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation" style=""><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">for</span> <span class="cm-variable">folder</span>, <span class="cm-variable">current_class</span> <span class="cm-keyword">in</span> <span class="cm-builtin">zip</span>(<span class="cm-variable">folders</span>, <span class="cm-builtin">range</span>(<span class="cm-number">2</span>)):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-builtin">print</span>(<span class="cm-string">'[Info] loading images from "'</span> <span class="cm-operator">+</span> <span class="cm-variable">folder</span> <span class="cm-operator">+</span> <span class="cm-string">'".........'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">images_path</span> <span class="cm-operator">=</span> <span class="cm-builtin">sorted</span>(<span class="cm-builtin">list</span>(<span class="cm-variable">paths</span>.<span class="cm-property">list_images</span>(<span class="cm-variable">dataSet_path</span> <span class="cm-operator">+</span> <span class="cm-string">"/"</span> <span class="cm-operator">+</span> <span class="cm-variable">folder</span>)))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-keyword">for</span> <span class="cm-variable">image_path</span> <span class="cm-keyword">in</span> <span class="cm-variable">images_path</span>:</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-comment"># data_ordinaire list</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">image</span> <span class="cm-operator">=</span> <span class="cm-variable">cv2</span>.<span class="cm-property">imread</span>(<span class="cm-variable">image_path</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">image</span> <span class="cm-operator">=</span> <span class="cm-variable">cv2</span>.<span class="cm-property">resize</span>(<span class="cm-variable">image</span>, (<span class="cm-number">64</span>, <span class="cm-number">64</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">data_list</span>.<span class="cm-property">append</span>(<span class="cm-variable">image</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-comment"># labels list</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">labels_list</span>.<span class="cm-property">append</span>(<span class="cm-variable">folder</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-builtin">print</span>(<span class="cm-string">'[Info] loading from        "'</span> <span class="cm-operator">+</span> <span class="cm-variable">folder</span> <span class="cm-operator">+</span> <span class="cm-string">'" '</span> <span class="cm-operator">+</span> <span class="cm-builtin">str</span>(<span class="cm-variable">np</span>.<span class="cm-property">size</span>(<span class="cm-variable">images_path</span>)) <span class="cm-operator">+</span> <span class="cm-string">' images'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-keyword">if</span> <span class="cm-variable">current_class</span> <span class="cm-operator">==</span> <span class="cm-number">1</span>:</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-builtin">print</span>(<span class="cm-string">'[Info] training network for 2 classes....'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">data_list</span> <span class="cm-operator">=</span> <span class="cm-variable">np</span>.<span class="cm-property">array</span>(<span class="cm-variable">data_list</span>, <span class="cm-variable">dtype</span><span class="cm-operator">=</span><span class="cm-string">"float"</span>) <span class="cm-operator">/</span> <span class="cm-number">255.0</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">labels_list</span> <span class="cm-operator">=</span> <span class="cm-variable">np</span>.<span class="cm-property">array</span>(<span class="cm-variable">labels_list</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-comment"># split data</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        (<span class="cm-variable">trainX</span>, <span class="cm-variable">testX</span>, <span class="cm-variable">trainY</span>, <span class="cm-variable">testY</span>) <span class="cm-operator">=</span> <span class="cm-variable">train_test_split</span>(<span class="cm-variable">data_list</span>, <span class="cm-variable">labels_list</span>, <span class="cm-variable">test_size</span><span class="cm-operator">=</span><span class="cm-number">0.25</span>, <span class="cm-variable">random_state</span><span class="cm-operator">=</span><span class="cm-number">42</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">data_list</span>, <span class="cm-variable">labels_list</span> <span class="cm-operator">=</span> [], []</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">trainY</span> <span class="cm-operator">=</span> <span class="cm-variable">lb</span>.<span class="cm-property">fit_transform</span>(<span class="cm-variable">trainY</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">trainY</span> <span class="cm-operator">=</span> <span class="cm-variable">np</span>.<span class="cm-property">hstack</span>((<span class="cm-number">1</span> <span class="cm-operator">-</span> <span class="cm-variable">trainY</span>, <span class="cm-variable">trainY</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">testY</span> <span class="cm-operator">=</span> <span class="cm-variable">lb</span>.<span class="cm-property">transform</span>(<span class="cm-variable">testY</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">testY</span> <span class="cm-operator">=</span> <span class="cm-variable">np</span>.<span class="cm-property">hstack</span>((<span class="cm-number">1</span> <span class="cm-operator">-</span> <span class="cm-variable">testY</span>, <span class="cm-variable">testY</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">model</span> <span class="cm-operator">=</span> <span class="cm-variable">build_model</span>(<span class="cm-variable">classes_size</span><span class="cm-operator">=</span><span class="cm-builtin">len</span>(<span class="cm-variable">lb</span>.<span class="cm-property">classes_</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">model</span>.<span class="cm-property">compile</span>(<span class="cm-variable">loss</span><span class="cm-operator">=</span><span class="cm-string">"categorical_crossentropy"</span>, <span class="cm-variable">optimizer</span><span class="cm-operator">=</span><span class="cm-variable">opt</span>, <span class="cm-variable">metrics</span><span class="cm-operator">=</span>[<span class="cm-string">"accuracy"</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-comment"># train the neural network</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">H</span> <span class="cm-operator">=</span> <span class="cm-variable">model</span>.<span class="cm-property">fit</span>(<span class="cm-variable">trainX</span>, <span class="cm-variable">trainY</span>, <span class="cm-variable">validation_data</span><span class="cm-operator">=</span>(<span class="cm-variable">testX</span>, <span class="cm-variable">testY</span>), <span class="cm-variable">epochs</span><span class="cm-operator">=</span><span class="cm-variable">EPOCHS</span>, <span class="cm-variable">batch_size</span><span class="cm-operator">=</span><span class="cm-variable">batch_size</span>, <span class="cm-variable">verbose</span><span class="cm-operator">=</span><span class="cm-number">2</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-comment"># evaluate the network</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-builtin">print</span>(<span class="cm-string">"[INFO] evaluating network..."</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">predictions</span> <span class="cm-operator">=</span> <span class="cm-variable">model</span>.<span class="cm-property">predict</span>(<span class="cm-variable">testX</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">rp</span> <span class="cm-operator">=</span> <span class="cm-variable">classification_report</span>(<span class="cm-variable">testY</span>.<span class="cm-property">argmax</span>(<span class="cm-variable">axis</span><span class="cm-operator">=</span><span class="cm-number">1</span>), <span class="cm-variable">predictions</span>.<span class="cm-property">argmax</span>(<span class="cm-variable">axis</span><span class="cm-operator">=</span><span class="cm-number">1</span>), <span class="cm-variable">target_names</span><span class="cm-operator">=</span><span class="cm-variable">lb</span>.<span class="cm-property">classes_</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-builtin">print</span>(<span class="cm-variable">rp</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 657px;"></div><div class="CodeMirror-gutters" style="display: none; height: 671px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout"><pre>[Info] loading images from "Potato_Early_blight".........
[Info] loading from        "Potato_Early_blight" 1000 images
[Info] loading images from "Potato_Healthy".........
[Info] loading from        "Potato_Healthy" 152 images
[Info] training network for 2 classes....
Train on 864 samples, validate on 288 samples
Epoch 1/5
 - 13s - loss: 0.4234 - accuracy: 0.8345 - val_loss: 0.3198 - val_accuracy: 0.9062
Epoch 2/5
 - 9s - loss: 0.2854 - accuracy: 0.8553 - val_loss: 0.1945 - val_accuracy: 0.9062
Epoch 3/5
 - 9s - loss: 0.2038 - accuracy: 0.8889 - val_loss: 0.1565 - val_accuracy: 0.9688
Epoch 4/5
 - 9s - loss: 0.1555 - accuracy: 0.9421 - val_loss: 0.0341 - val_accuracy: 0.9931
Epoch 5/5
 - 9s - loss: 0.0605 - accuracy: 0.9757 - val_loss: 0.0069 - val_accuracy: 1.0000
[INFO] evaluating network...
                     precision    recall  f1-score   support

Potato_Early_blight       1.00      1.00      1.00       261
     Potato_Healthy       1.00      1.00      1.00        27

           accuracy                           1.00       288
          macro avg       1.00      1.00      1.00       288
       weighted avg       1.00      1.00      1.00       288

</pre></div></div></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[5]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.59998px; left: 224px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 344.6px; margin-bottom: -16px; border-right-width: 14px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style="visibility: hidden;"><div class="CodeMirror-cursor" style="left: 224px; top: 0px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-builtin">print</span>(<span class="cm-string">"=\n=\n=\n=\nMETHOD \n=\n=\n=\n="</span>)</span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 28px;"></div><div class="CodeMirror-gutters" style="display: none; height: 42px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout"><pre>=
=
=
=
METHOD 
=
=
=
=
</pre></div></div></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[6]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.59998px; left: 5.60001px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true" style="bottom: 0px;"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 756.2px; margin-bottom: -16px; border-right-width: 14px; min-height: 334px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors"><div class="CodeMirror-cursor" style="left: 5.60001px; top: 0px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation" style=""><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">def</span> <span class="cm-def">features_extractor</span>(<span class="cm-variable">model</span>, <span class="cm-variable">x</span>):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">mod</span> <span class="cm-operator">=</span> <span class="cm-variable">model</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-comment">#mod = mod.layers.pop()</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">mod</span> <span class="cm-operator">=</span> <span class="cm-variable">Model</span>(<span class="cm-variable">inputs</span><span class="cm-operator">=</span><span class="cm-variable">mod</span>.<span class="cm-property">inputs</span>, <span class="cm-variable">outputs</span><span class="cm-operator">=</span><span class="cm-variable">mod</span>.<span class="cm-property">layers</span>[<span class="cm-operator">-</span><span class="cm-number">4</span>].<span class="cm-property">output</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    </span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-keyword">return</span> <span class="cm-variable">mod</span>.<span class="cm-property">predict</span>(<span class="cm-variable">x</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">def</span> <span class="cm-def">finding_neighbours</span>(<span class="cm-variable">center</span>, <span class="cm-variable">data_pos</span>, <span class="cm-variable">coordinates</span>, <span class="cm-variable">n_samples</span>):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">distances</span> <span class="cm-operator">=</span> []</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-keyword">for</span> <span class="cm-variable">xy</span> <span class="cm-keyword">in</span> <span class="cm-variable">coordinates</span>:</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-comment"># Phitagors theory</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">distances</span>.<span class="cm-property">append</span>(<span class="cm-variable">np</span>.<span class="cm-property">sqrt</span>(<span class="cm-variable">np</span>.<span class="cm-property">square</span>(<span class="cm-variable">xy</span>[<span class="cm-number">1</span>]<span class="cm-operator">-</span><span class="cm-variable">center</span>[<span class="cm-number">1</span>])<span class="cm-operator">+</span> <span class="cm-variable">np</span>.<span class="cm-property">square</span>(<span class="cm-variable">xy</span>[<span class="cm-number">0</span>]<span class="cm-operator">-</span><span class="cm-variable">center</span>[<span class="cm-number">0</span>])))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">x_y</span> <span class="cm-operator">=</span> <span class="cm-builtin">list</span>(<span class="cm-builtin">zip</span>(<span class="cm-variable">data_pos</span>, <span class="cm-variable">coordinates</span>, <span class="cm-variable">distances</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-comment"># sort by distances</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">selected_data</span> <span class="cm-operator">=</span> <span class="cm-builtin">sorted</span>(<span class="cm-variable">x_y</span>, <span class="cm-variable">key</span><span class="cm-operator">=</span><span class="cm-keyword">lambda</span> <span class="cm-variable">sample</span>: <span class="cm-variable">sample</span>[<span class="cm-number">2</span>], <span class="cm-variable">reverse</span><span class="cm-operator">=</span><span class="cm-keyword">False</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">selected_data</span> <span class="cm-operator">=</span> <span class="cm-variable">np</span>.<span class="cm-property">asarray</span>(<span class="cm-variable">selected_data</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">n_samples</span> <span class="cm-operator">=</span> <span class="cm-builtin">int</span>(<span class="cm-variable">n_samples</span> <span class="cm-operator">*</span> <span class="cm-builtin">len</span>(<span class="cm-variable">selected_data</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">selected_data</span> <span class="cm-operator">=</span> (<span class="cm-variable">selected_data</span>[:<span class="cm-variable">n_samples</span>, <span class="cm-number">0</span>:<span class="cm-number">2</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-keyword">return</span> <span class="cm-variable">selected_data</span></span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 334px;"></div><div class="CodeMirror-gutters" style="display: none; height: 348px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[7]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.60001px; left: 5.60001px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 395px; margin-bottom: -16px; border-right-width: 14px; min-height: 181px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors"><div class="CodeMirror-cursor" style="left: 5.60001px; top: 0px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation" style=""><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">x1</span>, <span class="cm-variable">x2</span><span class="cm-operator">=</span> [], []</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">for</span> <span class="cm-variable">i</span> <span class="cm-keyword">in</span> <span class="cm-builtin">range</span>(<span class="cm-builtin">len</span>(<span class="cm-variable">trainX</span>)):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-keyword">if</span> <span class="cm-variable">trainY</span>[<span class="cm-variable">i</span>,<span class="cm-number">0</span>] <span class="cm-operator">==</span> <span class="cm-number">1</span>:</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-comment"># data class 1</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">x1</span>.<span class="cm-property">append</span>(<span class="cm-variable">trainX</span>[<span class="cm-variable">i</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-keyword">else</span>:</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-comment"># data claas 2</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-variable">x2</span>.<span class="cm-property">append</span>(<span class="cm-variable">trainX</span>[<span class="cm-variable">i</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">fs_1</span> <span class="cm-operator">=</span> <span class="cm-variable">features_extractor</span>(<span class="cm-variable">model</span>, <span class="cm-variable">np</span>.<span class="cm-property">array</span>(<span class="cm-variable">x1</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">fs_2</span> <span class="cm-operator">=</span> <span class="cm-variable">features_extractor</span>(<span class="cm-variable">model</span>, <span class="cm-variable">np</span>.<span class="cm-property">array</span>(<span class="cm-variable">x2</span>))</span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 181px;"></div><div class="CodeMirror-gutters" style="display: none; height: 195px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[8]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 175.6px; left: 366.8px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 714.2px; margin-bottom: -16px; border-right-width: 14px; min-height: 198px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style="visibility: hidden;"><div class="CodeMirror-cursor" style="left: 366.8px; top: 170px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation" style=""><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">mid</span> <span class="cm-operator">=</span> <span class="cm-builtin">int</span>(<span class="cm-variable">fs_1</span>.<span class="cm-property">shape</span>[<span class="cm-number">1</span>]<span class="cm-operator">/</span><span class="cm-number">2</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">coor_features_1</span> <span class="cm-operator">=</span> []</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">for</span> <span class="cm-variable">i</span> <span class="cm-keyword">in</span> <span class="cm-builtin">range</span>(<span class="cm-builtin">len</span>(<span class="cm-variable">fs_1</span>)):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">coor_features_1</span>.<span class="cm-property">append</span>([<span class="cm-variable">np</span>.<span class="cm-property">sum</span>(<span class="cm-variable">fs_1</span>[<span class="cm-variable">i</span>,<span class="cm-number">0</span>:<span class="cm-variable">mid</span>])<span class="cm-operator">/</span><span class="cm-variable">mid</span>, <span class="cm-variable">np</span>.<span class="cm-property">sum</span>(<span class="cm-variable">fs_1</span>[<span class="cm-variable">i</span>, <span class="cm-variable">mid</span><span class="cm-operator">+</span><span class="cm-number">1</span>:])<span class="cm-operator">/</span><span class="cm-variable">mid</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">coor_features_1</span> <span class="cm-operator">=</span> <span class="cm-variable">np</span>.<span class="cm-property">array</span>(<span class="cm-variable">coor_features_1</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">coor_features_2</span> <span class="cm-operator">=</span> []</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">for</span> <span class="cm-variable">i</span> <span class="cm-keyword">in</span> <span class="cm-builtin">range</span>(<span class="cm-builtin">len</span>(<span class="cm-variable">fs_2</span>)):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">coor_features_2</span>.<span class="cm-property">append</span>([<span class="cm-variable">np</span>.<span class="cm-property">sum</span>(<span class="cm-variable">fs_2</span>[<span class="cm-variable">i</span>,<span class="cm-number">0</span>:<span class="cm-variable">mid</span>]<span class="cm-operator">/</span><span class="cm-variable">mid</span>), <span class="cm-variable">np</span>.<span class="cm-property">sum</span>(<span class="cm-variable">fs_2</span>[<span class="cm-variable">i</span>, <span class="cm-variable">mid</span><span class="cm-operator">+</span><span class="cm-number">1</span>:])<span class="cm-operator">/</span><span class="cm-variable">mid</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">coor_features_2</span> <span class="cm-operator">=</span> <span class="cm-variable">np</span>.<span class="cm-property">array</span><span class=" CodeMirror-matchingbracket">(</span><span class="cm-variable">coor_features_2</span><span class=" CodeMirror-matchingbracket">)</span></span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 198px;"></div><div class="CodeMirror-gutters" style="display: none; height: 212px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[9]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.59998px; left: 341.6px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 344.6px; margin-bottom: -16px; border-right-width: 14px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style="visibility: hidden;"><div class="CodeMirror-cursor" style="left: 341.6px; top: 0px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-builtin">print</span><span class=" CodeMirror-matchingbracket">(</span><span class="cm-string">"=\n=\n=\n=\nclass 2\n=\n=\n=\n="</span><span class=" CodeMirror-matchingbracket">)</span></span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 28px;"></div><div class="CodeMirror-gutters" style="display: none; height: 42px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout"><pre>=
=
=
=
class 2
=
=
=
=
</pre></div></div></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[10]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 22.6px; left: 156.8px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 705.8px; margin-bottom: -16px; border-right-width: 14px; min-height: 79px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style="visibility: hidden;"><div class="CodeMirror-cursor" style="left: 156.8px; top: 17px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">import</span> <span class="cm-variable">matplotlib</span>.<span class="cm-property">pyplot</span> <span class="cm-keyword">as</span> <span class="cm-variable">plt</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-operator">%</span><span class="cm-variable">matplotlib</span> <span class="cm-variable">inline</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">plt</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">coor_features_1</span>[:,<span class="cm-number">0</span>],<span class="cm-variable">coor_features_1</span>[:,<span class="cm-number">1</span>], <span class="cm-variable">label</span><span class="cm-operator">=</span><span class="cm-string">"Features of class 1"</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">plt</span>.<span class="cm-property">legend</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 79px;"></div><div class="CodeMirror-gutters" style="display: none; height: 93px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt output_prompt"><bdi>Out[10]:</bdi></div><div class="output_subarea output_text output_result"><pre>&lt;matplotlib.legend.Legend at 0x1d7e9709608&gt;</pre></div></div><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_png"><img src="data:image/png;base64,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
"></div></div></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[17]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 175.6px; left: 560px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true" style="bottom: 0px;"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 966.2px; margin-bottom: -16px; border-right-width: 14px; min-height: 351px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style="visibility: hidden;"><div class="CodeMirror-cursor" style="left: 560px; top: 170px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation" style=""><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">cluster</span> <span class="cm-keyword">import</span> <span class="cm-variable">KMeans</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">fig</span>, <span class="cm-variable">ax</span> <span class="cm-operator">=</span> <span class="cm-variable">plt</span>.<span class="cm-property">subplots</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">kmeans</span> <span class="cm-operator">=</span> <span class="cm-variable">KMeans</span>(<span class="cm-variable">n_clusters</span><span class="cm-operator">=</span><span class="cm-number">4</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">kmeans</span>.<span class="cm-property">fit</span>(<span class="cm-variable">coor_features_1</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">num_centers</span> <span class="cm-operator">=</span> <span class="cm-builtin">len</span>(<span class="cm-variable">kmeans</span>.<span class="cm-property">cluster_centers_</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">centers_center</span> <span class="cm-operator">=</span> [<span class="cm-builtin">sum</span>(<span class="cm-variable">kmeans</span>.<span class="cm-property">cluster_centers_</span>[:,<span class="cm-number">0</span>])<span class="cm-operator">/</span><span class="cm-variable">num_centers</span>, <span class="cm-builtin">sum</span>(<span class="cm-variable">kmeans</span>.<span class="cm-property">cluster_centers_</span>[:,<span class="cm-number">1</span>])<span class="cm-operator">/</span><span class="cm-variable">num_centers</span>]</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-comment"># plot clusters</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">scatter</span> <span class="cm-operator">=</span> <span class="cm-variable">ax</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">coor_features_1</span>[:,<span class="cm-number">0</span>],<span class="cm-variable">coor_features_1</span>[:,<span class="cm-number">1</span>], <span class="cm-variable">c</span><span class="cm-operator">=</span><span class="cm-variable">kmeans</span>.<span class="cm-property">labels_</span>, <span class="cm-variable">cmap</span><span class="cm-operator">=</span><span class="cm-string">'rainbow'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">legend1</span> <span class="cm-operator">=</span> <span class="cm-variable">ax</span>.<span class="cm-property">legend</span>(<span class="cm-operator">*</span><span class="cm-variable">scatter</span>.<span class="cm-property">legend_elements</span>(), <span class="cm-variable">loc</span><span class="cm-operator">=</span><span class="cm-string">"lower right"</span>, <span class="cm-variable">title</span><span class="cm-operator">=</span><span class="cm-string">"groups"</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">ax</span>.<span class="cm-property">add_artist</span>(<span class="cm-variable">legend1</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-comment"># plot centers center</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">ax</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">centers_center</span>[<span class="cm-number">0</span>], <span class="cm-variable">centers_center</span>[<span class="cm-number">1</span>], <span class="cm-variable">cmap</span><span class="cm-operator">=</span><span class="cm-string">'rainbow'</span>, <span class="cm-variable">label</span><span class="cm-operator">=</span><span class="cm-string">"Centers center"</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-comment"># plot centers</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">ax</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">kmeans</span>.<span class="cm-property">cluster_centers_</span>[:,<span class="cm-number">0</span>], <span class="cm-variable">kmeans</span>.<span class="cm-property">cluster_centers_</span>[:,<span class="cm-number">1</span>],<span class="cm-variable">cmap</span><span class="cm-operator">=</span><span class="cm-string">'rainbow'</span>, <span class="cm-variable">marker</span><span class="cm-operator">=</span><span class="cm-string">"+"</span>, <span class="cm-variable">label</span><span class="cm-operator">=</span><span class="cm-string">"centers"</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">for</span> <span class="cm-variable">n</span> <span class="cm-keyword">in</span> <span class="cm-builtin">range</span>(<span class="cm-builtin">len</span>(<span class="cm-variable">kmeans</span>.<span class="cm-property">cluster_centers_</span>)):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-builtin">print</span>(<span class="cm-string">"cluster"</span>,<span class="cm-variable">n</span> ,<span class="cm-string">"contains"</span>, <span class="cm-builtin">sum</span>(<span class="cm-variable">kmeans</span>.<span class="cm-property">labels_</span><span class="cm-operator">==</span><span class="cm-variable">n</span>), <span class="cm-string">"features"</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-builtin">print</span>(<span class="cm-string">"center of centers : X="</span>,<span class="cm-variable">centers_center</span>[<span class="cm-number">0</span>],<span class="cm-string">"Y="</span>, <span class="cm-variable">centers_center</span>[<span class="cm-number">1</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">plt</span>.<span class="cm-property">legend</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 351px;"></div><div class="CodeMirror-gutters" style="display: none; height: 365px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout"><pre>cluster 0 contains 229 features
cluster 1 contains 148 features
cluster 2 contains 136 features
cluster 3 contains 226 features
center of centers : X= 0.07840379705024364 Y= 0.07301131249134153
</pre></div></div><div class="output_area"><div class="run_this_cell"></div><div class="prompt output_prompt"><bdi>Out[17]:</bdi></div><div class="output_subarea output_text output_result"><pre>&lt;matplotlib.legend.Legend at 0x1d7ff886cc8&gt;</pre></div></div><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_png"><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYwAAAD4CAYAAAD//dEpAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAgAElEQVR4nOydd5hcZdm472fOtJ3tNbvJpgJSAiGQ0ES6EiQqIFUQCfqBIIjoTxS7gCifDQVFKQKKCFIEaRpEQvnoCYROCKRu7zu9nuf3x8wus7szu7PJpmx47+uaKzOnvOc9M5v3OU8XVcVgMBgMhrFwbOsJGAwGg2FyYASGwWAwGArCCAyDwWAwFIQRGAaDwWAoCCMwDAaDwVAQzm09gfFQU1Ojs2bN2tbTMBgMhknFihUrulS1dnPHmVQCY9asWSxfvnxbT8NgMBgmFSKyfiLGMSYpg8FgMBSEERgGg8FgKAgjMAwGg8FQEJPKh5GLRCJBU1MT0Wh0W0/FUCBer5fGxkZcLte2norBYBgHk15gNDU1UVpayqxZsxCRbT0dwxioKt3d3TQ1NTF79uxtPR2DwTAOJr1JKhqNUl1dbYTFJEFEqK6uNhqhwTAJmfQCAzDCYpJhfi+DYXJSkMAQkWNEZJWIvCcil+bYf6iIvCwiSRE5KWv7ESKyMusVFZHjM/tuFZG1WfvmT9xtGQwGg2GiGVNgiIgF/B74JLAH8DkR2WPYYRuAJcDfsjeq6jJVna+q84EjgTDwaNYhlwzsV9WVm34b25a2tjZOO+00dtppJ/bYYw+OPfZY3n333U0a69Zbb6WlpWWCZzhxrFu3jr/97W9jH2gwGHY4CtEw9gfeU9U1qhoH7gSOyz5AVdep6muAPco4JwH/UtXwJs92O0RVOeGEEzj88MN5//33eeutt/jpT39Ke3v7Jo23KQIjmUxu0rU2hU0RGKlUagvNxmAwbE0KERjTgI1Zn5sy28bLacAdw7ZdKSKvicjVIuLJdZKInCsiy0VkeWdn5yZcdij3v9LMwVc9zuxLH+bgqx7n/leaN2u8ZcuW4XK5OO+88wa3zZ8/n0MOOQSAX/ziF+y3337MmzePH/3oR0B60d19990555xzmDt3LkcffTSRSIR77rmH5cuXc8YZZzB//nwikQgrVqzgsMMOY8GCBSxatIjW1lYADj/8cL773e9y2GGH8dvf/pa7776bPffck7333ptDDz0051x//vOfs9dee7H33ntz6aVpy+L777/PMcccw4IFCzjkkEN45513AFiyZAkXXXQRH/3oR5kzZw733HMPAJdeeilPP/008+fP5+qrryaVSnHJJZcM3uP1118PwBNPPMERRxzB6aefzl577bVZ37HBYNhOUNVRX8DJwE1Zn88Ers1z7K3ASTm2NwCdgGvYNgE8wJ+BH441lwULFuhw3nrrrRHb8nHfy0262/f/pTO//dDga7fv/0vve7mp4DGG89vf/lYvvvjinPuWLl2q55xzjtq2ralUShcvXqxPPvmkrl27Vi3L0ldeeUVVVU8++WS97bbbVFX1sMMO05deeklVVePxuB500EHa0dGhqqp33nmnnn322YPHnX/++YPX2nPPPbWpKX0fvb29I+byyCOP6EEHHaShUEhVVbu7u1VV9cgjj9R3331XVVWff/55PeKII1RV9ayzztKTTjpJU6mUvvnmm7rTTjupquqyZct08eLFg+Nef/31esUVV6iqajQa1QULFuiaNWt02bJl6vP5dM2aNTm/m/H8bgaDYfMAlusY62shr0LyMJqA6VmfG4HxGtlPAe5T1USWoGrNvI2JyC3AN8c55rj5xdJVRBJDzSORRIpfLF3F8ftsitI0Oo8++iiPPvoo++yzDwDBYJDVq1czY8YMZs+ezfz5aT//ggULWLdu3YjzV61axRtvvMEnPvEJIG3aaWhoGNx/6qmnDr4/+OCDWbJkCaeccgqf/exnR4z12GOPcfbZZ+Pz+QCoqqoiGAzy7LPPcvLJJw8eF4vFBt8ff/zxOBwO9thjj7wmtkcffZTXXnttUAPp7+9n9erVuN1u9t9/f5NrYTDsQBQiMF4CdhGR2UAzadPS6eO8zueA72RvEJEGVW2VdIzl8cAb4xxz3LT0Rca1vRDmzp07uFgOR1X5zne+w5e//OUh29etW4fH84EFzrIsIpGRc1BV5s6dy3PPPZdz/OLi4sH3f/zjH3nhhRd4+OGHmT9/PitXrqS6unrIWMPDWW3bpqKigpUrc8cbZM8x/ZCS+x6vvfZaFi1aNGT7E088MWR+BoNh8jOmD0NVk8CFwFLgbeAuVX1TRC4Xkc8AiMh+ItJE2nx1vYi8OXC+iMwiraE8OWzo20XkdeB1oAb4yebfzuhMrSga1/ZCOPLII4nFYtx4442D21566SWefPJJFi1axM0330wwGASgubmZjo6OUccrLS0lEAgAsOuuu9LZ2TkoMBKJBG+++WbO895//30OOOAALr/8cmpqati4ceOQ/UcffTQ333wz4XA65qCnp4eysjJmz57N3XffDaQX/1dffbXg+QEsWrSIP/zhDyQSaeXx3XffJRQKjTqGwWCYnBRUGkRVHwEeGbbth1nvXyJtqsp17jpyOMlV9cjxTHQiuGTRrnznH68PMUsVuSwuWbTrJo8pItx3331cfPHFXHXVVXi9XmbNmsVvfvMbdtllF95++20OOuggAEpKSvjrX/+KZVl5x1uyZAnnnXceRUVFPPfcc9xzzz1cdNFF9Pf3k0wmufjii5k7d+7Ie7vkElavXo2qctRRR7H33nsP2X/MMcewcuVKFi5ciNvt5thjj+WnP/0pt99+O+effz4/+clPSCQSnHbaaSPOzWbevHk4nU723ntvlixZwte+9jXWrVvHvvvui6pSW1vL/fffv4nfpsFg2J6RfKaG7ZGFCxfq8AZKb7/9NrvvvnvBY9z/SjO/WLqKlr4IUyuKuGTRrlvEf2EYnfH+bgaDYdMRkRWqunBzx5n0xQfHy/H7TDMCwmAwGDaBHaKWlMFgMBi2PB86DcNgMIyfmG3zRDhMn22zn9fLHLd7W0/JsA0wAsNgMIzK27EYX2ptJalKClDgUyUlXFZTYyoPf8gwJimDwZAXW5UL2trot21CqkRVianySDDI0h0gfDquytPhMMtCIUL2aKXw0mHniUkUJLQlMBqGwWDIyzvxOIEcC2lElbv9fo4pKdkGs5oYXopE+Gp7e7rsBZACLq+pYXFp6ZDjEqpc09PDnX4/EVVmuVx8v6aGA4s2PX9rsmI0jO2Avr4+rrvuum09DYNhBHFV8hmd4pP4aTtk23ylrY2AbRNUHdSeftDVxYZEYsixV3R18Te/n3BGsKxNJLigrY23ssrofFgwAmM7YFMEhqpij6FCGwyby1yPByuHn8IrwqcmsXaxLI85LaXKg1mVDPpSKR4MBIgOE44xVa7v7d2ic9we+XAKjFsWp18TxF/+8hfmzZvH3nvvzZlnnklnZycnnngi++23H/vttx/PPPMMAD/+8Y/54he/yOGHH86cOXO45pprgHTJ8Pfff5/58+dzySWXAKOXRf/KV77Cvvvuy8aNG1myZAl77rkne+21F1dfffWE3ZPBAOAS4ed1dXhFcGW2+USY6/FwQlnZNp3b5hBSzdm8JwlDTHCtySTuHAJTgdXDNJEPA8aHsZm8+eabXHnllTzzzDPU1NTQ09PDhRdeyNe//nU+9rGPsWHDBhYtWsTbb78NwDvvvMOyZcsIBALsuuuunH/++Vx11VW88cYbg0UAH330UVavXs2LL76IqvKZz3yGp556ihkzZrBq1SpuueUWrrvuOlasWEFzczNvvJGu29jX17fNvgfDjsshPh8PTZ/OPwMBulIpDi4q4lCfL6fmsT2yJh7nut5eXo3FmO50cl5lJQcVFZHLoFYkwuFZRTMbXS5yiQUHMPdDGFr84RIYA1rF+v8b+vnshzd5yMcff5yTTjqJmpoaIF02/LHHHuOtt94aPMbv9w8W7Fu8eDEejwePx0NdXV3OsuGjlUWfOXMmBx54IABz5sxhzZo1fPWrX2Xx4sUcffTRm3wfBsNoNGQW2snG6nic05ubiWY0ipZkklfb2vhJbS2fLyvjbxlHNqSFxcFFRRzg9Q6eX+pwcFpZGX/3+4eYpTwifHkSfh+by4dLYGwB8pUNf+655yjKEUUxvKx5rvaqo5VFzy4ZXllZyauvvsrSpUv5/e9/z1133cXNN9+8ubdkMOwwXN3dTSTjrB4gqspV3d0smzGDj/p83B8IEFNlcUkJR/p8I/4/f7OqilrL4s/9/fSlUuzl8fDtmhp2NhrGDs6AJjEBmsUARx11FCeccAJf//rXqa6upqenh6OPPprf/e53g/6IlStXDjZLykWukuE/+MEPOOOMMygpKaG5uRmXyzXivK6uLtxuNyeeeCI77bQTS5Ys2ez7MRh2JFbGYjlNT37bpieV4sCiojHDYx0inF1RwdkVFVtmkpOID5fA2ALMnTuX733vexx22GFYlsU+++zDNddcwwUXXMC8efNIJpMceuih/PGPf8w7RnV1NQcffDB77rknn/zkJ/nFL35RUFn05uZmzj777MFoqZ/97Gdb7kYNhgkgYts8GAjwVCRCvdPJaWVlBT2p96dSpICqUVoD5KLGsujPEU0opM1NhvHxoStvbtg+ML/bh4+QbXNaczOtySQRVSzSUVg/q63l6Dwhui3JJJd2dPBaNArALJeLq+rq2C3LtDuAqpIAXDBoVnooEODHXV2DfgpI+x+OKynhR7W1E32L2y0TVd7ciFiDwbBVuL2/n+ZEYnDxTsFgslyukhtJVb7Q3Mwr0SgJIEE6lPWs1lb6Uqkhx/7D7+fwDRvYd+1aDl2/njv7+9GMX+L8igqKRPCJ4BbhmOJivpMJUjGMD2OSMhgMW4VHQyFy5UarKu/EYuyVFZ0E8H/hMH7bHpEvkVTlgUCAL2R8Cg8GAlzZ3T0YxdRj2/yypwcBTi0v50uVlZxRXk5LMkmNZVE2TrOW4QMK0jBE5BgRWSUi74nIpTn2HyoiL4tIUkROGrYvJSIrM68HsrbPFpEXRGS1iPxdRDY55GAymdUM5vf6MNKbShEYphUMkAJKcvgTWpJJRsYQprWSDVnRhdf29o7IxI6ocl1WXpLX4WCO222ExWYypsAQEQv4PfBJYA/gcyKyx7DDNgBLgL/lGCKiqvMzr89kbf9f4GpV3QXoBb60CfPH6/XS3d1tFqFJgqrS3d2Nd9jTpGHHZU08zic3bqQ9h8BwAI1OJ7NzOL739HhyLlA+EeZn/f205QhNB+hKpbDNujChFGKS2h94T1XXAIjIncBxwGBmmqquy+wrqLiRpD1SRwKnZzb9Gfgx8IcC5z1IY2MjTU1NdHZ2jvdUwzbC6/XS2Ni4racxqXgzFuOpUIgih4NjSkqod04ea/JlXV0EbXtEeKuDdELg7+vrc563l8fDPI+HlbEYsczC7wJqLYujs/KRprtcrMtRpqPesnBMkmz0yUIhf3XTgI1Zn5uAA8ZxDa+ILCddpuUqVb0fqAb6VHXg0aApc50RiMi5wLkAM2bMGLHf5XIxe/bscUzHYJg8qCqXdXXxYDBIXBUncE1vLz+pqeHYYWW4t0dsVV6ORnPmQjiAf0+fnndRFxF+NWUKV3V381w4jGQc1l+prBxS3+n/VVVxSUfHELOUV4SvV1VN8N0YCvFh5Po1x6PnzciEc50O/EZEdhrPmKp6g6ouVNWFtR+iMDiDAeCFaJSHgsHB0hZx0pVSf5B5at/eESCf18ArMqoG8FAgwMc3bOCJUIioKklVji0poXyYH+LI4mJ+WVfHHJcLFzDT5eKntbV8KkugtieTXN3dzTmtrfymu5uOPGYsw+gUomE0AdOzPjcCLYVeQFVbMv+uEZEngH2Ae4EKEXFmtIxxjWkwfFh4OBAYkkMwgAU8Ew6zKEf+QtS26UqlqLUsPNs4OU1EOLakhEeCwSFF/NwiHDeKhrQ2HueHXV2DpigAVDm3tZUnZ87EO+y+jigu5ogsM1U278bjfL65mYQqcdKNk+7w+7l92rQPZXmPzaGQv6aXgF0yUU1u4DTggTHOAUBEKkXEk3lfAxwMvKVpD/UyYCCi6izgn+OdvMGwwyOSt4HR8O22Kr/u7ubg9es5vqmJj61fzx96e1FV/h0McnJTE0etX8/3Ojpo2Yqlub9bU8NuHs9gLoRXhHkez6gmo/uDQVI5BKUCT4bD47r+T7q6CGWEBaTzOUKqXNnVNa5xDAVoGKqaFJELgaWkH2xuVtU3ReRyYLmqPiAi+wH3AZXAp0XkMlWdC+wOXJ9xhjtI+zAGnOXfBu4UkZ8ArwB/mvC7Mxi2Ab2pFFFV6i1rRCG78fKZkhL+FQyO0DJs4GCfb8i263t7uX1YVdU/9fXxRjTKC9Ho4BgPBoM8Hg5zX2PjVnGelzgc3DF1Kq/HYqxLJNjZ7WaPHJna2fhTqZwhtXFVngmHOdjnyxmKOxxV5ZVMlviQ7cCKHNsNozPpS4MYDNsLHckk38yUsXCIUG1ZXFVby4LN6P2sqvy8p4e/+/3YqlgiKPDLujqOzDLBqCoHrl9fsF/DCZxWVrbdZjw/EQrxzY6OnOa4oowQvmbKFD46TGjmYuHatTnHKRbhxQ9JwIwpDWIwbEeoKktaW1mZKWMRU6UlmeTLbW20DnOwxlXpSiZJFvCwJiJ8u7qau6ZN42tVVVxcVcVvp0xhl2G29xTpWk2FkgRe3I6fsA/1+Vjg9Q4Kh2wiqkRU+Vp7O5EC7vmE0lKG6zMe4MRJEGW2vTF5grkNhu2YFdEonckkw1PTUqrc1d/P16qrsVW5preXv/b3Y5MugvfVykpOLy8fc/yd3W5ejkb5RXc3DtJ2+N3cbq6ZMoUapxOnCI1OJxtzRP8IuUMQp22GOUpVWRmL0Z5MsqfHQ2OO8vubg0OE6+rrWRoK8evublpzJP0J8GwkwlFZmtbySIT7AgESmYiqw3w+Tiot5bVolHfjcdwiJIADvF4uqKzkkWCQl6NRpjmdHFdaOu5quB82jMAwGCaA4VrEAHEYLGNxXW8vt/X3D/oYYqr8qqeHModjSAhoLl6KRPjfrHpJkE7mu7C9nTunpVOYvltTw9fb20fkI+zicrEqHh90+g5s/+Io/R3Cts3SUIj34nF2c7s5urh4MOKqM5nki62ttCWTCGnhdWxxMVfU1k5oopyVibB6Jhzm/mAw5zHxrHu9urt70IejwH9DIXwOByFVXKSF5t5eL5dWVVHncnF6pnJuWBWPCNf19nJzQ8OImlaGDzAmKYNhAtjL4xmhXUDa3r6/10tKlT9nCYsBoqr8oYBe7LnOTZJuQbo+E/F0qM/H9fX17O/1UmNZHOD1cnN9PTdNncohPh/uzHzKHQ6uqKlh3zwLY0siwTEbN3JlVxe39vdzeVcXizdupDOZxFbly21trEskCKumo49U+XcoxN1+/5j3oapsTCRYn0gUXM7nmJKSnKapJPDRjH9oQyLBbZl2qwOjRkkXIoypEsyUPn85GuW5aJQbe3vZmLkHSAvvsCrf6ugwZYZGwWgYBsMEMMvt5iifj8fD4cGF3UW64c+nS0uJZBbWXBSSRJbvGCfQnUoxM2MS2sPjocHpZGU0yopUiks7O/lxbS3X1NfTn0rRb9tMzZiw8vHjri56U6nBKrFhVeKpFD/u7GRVPJ7TPBRV5Ta/nwrL4p5AgJejUbwinFxayleqqkiq8m48zvc7O2nJaCbVlsWv6urGfKL/WFERR/h8LMt8txZp7eP71dWDSXzPFBhqG1HltoxJMJ5jf3sqRVsqRcMkKr2yNTHfisEwQVxVV8cdfj93+v1EbZujios5v7ISn8OBqlJpWXTmWGw/UkDy2CE+H6uHmZUg/ZS9a9b532hv54VodPC4DckkX2lr4++ZJLVyy6InlWJVLEad08lOw53nqjwfiYwsKQ48GYmMOse1iQTf6OgY/BxV5S/9/dwbCOC37RFhss3JJF9qbeXRGTOoGOY7SKlyfW8vf/X7Cdg2OzudfKG8HH8qRZll8emSkiEFC30OR96M8uEEbJvyPCG5mim/YsiN+W4MhgnCEuHz5eV8PocTW0S4pKqKH3Z1jfAxfKO6esyxzywv5x+BAH2p1KAw8IrwtaoqijOLX1MiwYvR6AhNJqHKjb29nFJWxoPBIP8MBHA7HCRV2dXt5rr6+sEFezQPxKYYamJAbJRIphTwcDDIGcO+s592dXF/piQKwLvJJO/29eER4ZzycmYNc7IfWVzMFQUk4jlIm7H28Hj43bCy6A7SwrvWaBd5MT4Mg2Ersbi0lF9PmcIebjflDgf7eb38qaGBBQU4WSssi380NrKkvJzd3G4+VlTENVOm8IWshbY5mSSXrpICHg6FOLetjbsDAeJA0LaJqvJ6LMYFbW2DxzpEOMLn22pPklHVEea2/lSK+7KERTYxVX7X18fea9dydksL78TSLZlKHQ6unTKFYhFKRCgWwWKoALQyx11YWcmpZWXs7/XiFcFNOiejxrL45ZQpW+xedwRM4p7BsIPQlUzyiY0b8/pKRuP8igouzJTq6E6l+HxzM92ZjPXcbY/S5AvZLRSfCL+oq+PwrNDYt2Ixzm5pIVjAffhE+EdjI9MzGkfUtnkuEmFDIsE1wzQIC5hiWbRn/DPVDgd9to1k7uP8igrO3UEr3JrEPYPBMIQap5PjS0rwbkJo6019fbRnnvSrLYuHpk/nF1OmsHAU7ccJzHa5aNxEE45HhJ3dbg4Zlq09zenM6ZDORUyVPw3rrHdEcTErY7ERgjMFtKRSpEgLua6MXyVB2gF+Q38/D+cJ3zWkMQLDsEOhmiIW7ySZCm3rqWwTflBTw0WVlTQ4nZQ4HAWblizgqaxII0uE2S4XXXnaqjqAU0pLuWvaNC6orMw7rsCQLGtH5lqznE7Or6jgloYGrGECrtyyOKGkZER2di5SpPNRhvN2LDbCcT8WEVVu6O0d51kfLox3x7DD0OtfSVvPv0EV1RTFRXNonPJZLMeHJxHLIcJZFRWclUnK+1pbG4+Hw2Mung4RPFkLd8i2Ob25mb4cDmsn6Z4T36mpwZFJrhuoCDscAU4vL+eRYJCYKkf6fFxUVTWmY/l7NTXUWBY39/UxWmyWA0aUSQHYye2mKZkct7msO4+ANKQxGoZhhyAUWUdr9yPYdgxb4ygpQpE1bGy/e4tetzuVYmU0ut0uNN+orsaXpWnkM1bZMKSfxA09PfTlaKsKsL/Xy21Tpw5mdTtF+EyOvhyQFi5VlsXjM2fyzKxZXFFXV1AUkiXCV6qqWD5nDpfX1ORdqNx5MtbPq6gY99OwwJBe4YaRGA3DsEPQ1fcMqkN7PCgpwtENJJJ+XM6yCb1eUpVvt7fzaDg8uKgu8Hi4qaEBV1aMf2cyyX9CIeKqHOrzMWcrN+yZ6XJxf2Mjt/b18XI0ygynk6gqT0ciOGGw+u2vpkyhNDPvjmSSW/3+nMLCBRxWXDyi691stxsXMLzLRpwPkg7jmb4cL2VqN51YWlqQ8LgvEMipITmBy2prczZBeiocHiEchfQTci7R7iAdpnzxDur0niiMwDDsECSSuctSCBbJVHDCBcYvu7v597Ds4uWxGF9sbeW2TG2nfweDfLezE2Cw8ODJJSXUuVysiceZ7/GwuLQU3xbuitfgdHJpdTU39PVxbW8vSnrxTAFzXC5uamgYkjh3dx5hAflNQAu8XiwREsPMUj4R9i8qImDbfK65mbZkkogqbtK9Om5oaGCfMZ7qX8/ho4B0MuEPOzq41+/nd/X1FDsc9KVS3NLXx5/6+0fcg0uEXV0u1iWTpFTZ2+Mhrkp7KsV8r5fzKiqGJAMaRmIEhmGHoLhoFrFEFwx7FlVsPK6J7/lwZ566SS/HYvhTKRT4bmfniBajfw0EcJN+8l4aCvHHvj7+Pm0aNVs4Wezn3d38JWvOSnrBXRWP80ggwOlZZp11iUTeUNpqy2L/HAv8bh4Ph/t8/DcUGqJlFIuwp9vNlZ2dNCUSg/vipDWOb3d0sHT69FEbTRU7HPTnSf6LASujUX7c2cmSigqWtLQMqSeVTVwVlwjPz5qV91qG0TE+DMMOQU3Fx7AcHrL/pEVc1FUejsMx8U+NozU4XR2P83Q4nLdUxUDIaESVzlSK3/b0TPDshtKdSnFHHgFnAzf39w8e94feXt6Px3M+SVqkTVf5FvdTclTc7bFtjtq4kYeGCZLsuTWPUUvrjLKyUUOF48CjoRCnNTcTziMsIP2XYWpEbR4FCQwROUZEVonIeyJyaY79h4rIyyKSFJGTsrbPF5HnRORNEXlNRE7N2neriKwVkZWZ1/yJuSXDhxGXs5SdGs+jsnQfXM4KijzTaaz7LDUVH90i1yvLY0YSoM7pLDg6JwX8d5w9qsfLO7HYqHWWelIp3o/HWbxxIzf29bEqkRhR98krwuE+H/NGMR/d0Nc3QiikSAulfN+HTdpx/Ww4zPmtrZza1MQfenvTWpoqr0ej1DudHOD15sxiHyDJcN1yJG4RvjBKSXfD2IwpbkXEAn4PfAJoAl4SkQeyenMDbACWAN8cdnoY+IKqrhaRqcAKEVmqqgOZNpeo6j2bexMGA4DLWcbU2k9tlWt9q6qK7+eoXbS72810l4tSh2PUDOls3MOenhOq/DMQ4J+BAE4RTiwr49ji4nH1mojaNiFVqhwOpjido84lBpzR3Ewwx9O5i7QAPLWsbEgZklwMlFkvFAE+4nLxSDDItVlZ2e8mEtzr91PrdLI6Hh/MJt/J7aZIhFdisRHzHCvjvFiEH9XUsOcYvcRHI2zbJFUp28QmSz2pFO3JJDNcrsH6X5ONQvSz/YH3VHUNgIjcCRwHDAoMVV2X2TdEyKvqu1nvW0SkA6gFxm4AYDBsx5xQVkbQtvl1Tw8J0gvWvl4v12ZqEVVYFj+qqeGyri5s1cGn7IHXAB7gs1mmHFuV89vaWBmNDvahfj0W4+lwmP+tqxtzXlHb5oquLh4JhUCVCsvi+9XV7OZ283o8f/50YJQyHI/OmDHmde/x+3NW4h0NBT5TUsKvenuH+HoGHNEdmazsAd6Lxznc56PU4Y2XZwsAACAASURBVCBk20P2jSYs9nS7+eu0abg2sblTZzLJ9zo7eSFTrXdnt5sra2vZrUDhE7Ntvt/ZyWPhMC7S2tBZ5eVcVFk5qu9me6QQgTEN2Jj1uQk4YLwXEpH9ATfwftbmK0Xkh8B/gUtVNXc4hMGwHXJmRQWnl5fTnExS5nCMKNF9XGkp+xUVsTSTtDbf4+GK7u50I6LMMfO9Xs7LypR+PhLh1SxhAWlfx2OhEO/EYiMWqYFacAMLz3c6O3kyFBr0k3SkUnyrs5PfTpnCb3t6eGsUoZGL4doPgN5yLH7b5qbjb6fGsih1OPhZd/cIM1YhXNfXh1OV4f/xc5mXEqRLrP+pvp6zWlsLGn+608kfGho2WVjYmV7tTVlmunficc5qaeFfM2YU1NL1Z93d/Dfzmwx8+7f19zPN6eSksomN3tvSFCIwcn3T40qgFJEG4DbgLFUd+Fv4DtBGWojcAHwbuDzHuecC5wLMKOBJx2DYmlgizBiln/VUp5Ozs+zmDxYV8VI0SlMiwW4eD3OHCYDnIpHBLnDZpFR5MRodFBhvx2Jc0dXFa7EYXhFOLC3lC+XlPBEOj6jDFFPlTr+fuxsbeTMa5X+7u1mRJ1Q1Gwewj8fDi5EIq+NxGl0u9vN62RCPE7Jtbu7vxyNCfBRHs5u0HyOf7hGy7YL7WEA6/+WFSCRvAqKb9IJlA0f6fKM66QvhxWiUjmRyhDBMAPcHAqO2uYW0tvRAMDhCIEYyNbB2RIHRBEzP+twItBR6AREpAx4Gvq+qzw9sV9WBR4SYiNzCSP/HwHE3kBYoLFy4cPKU1jUYcuAQ4YCiIg7ItBYdTrVl4YERC4wAL4TDVFsWc91uzmxuHiyZEVHlLr+ft2OxwZDdbJR0C1OAuV4vZ1ZUsKK9fcy52sAz0SjPtLbiAm5a+j+8g7Bve7pi9C1L/weAsxfdlPN8N3D7tGnc7fdzf6asei5KHA6io/TMGEBIZ5nHye/grrIsPldezlE+34TkVDQlEjmFYUyVtQX4bMK2jZ3H3NdbwD1vbxTieXkJ2EVEZouIGzgNeKCQwTPH3wf8RVXvHravIfOvAMcDb4xn4gbDZGFtPM7Purq4oK2N2/v7CY2yUCwuKcnp3I4DT0Qi/Lizk5OzhEX2/tdyVGiFdDhsdsmL5lHMUsOf9gd8LnHSWsJ4qjMVOxzs5nbzo9pavltdPeLp1AIO9/ny1qByw2A4rVeEUoeDH9bW8vHi4iF1r7LpSKW42+8fLHe+uezu8eS8Y58I8wvwYZQ7HDlzbATYdzMc8NuKMTUMVU2KyIXAUtK/8c2q+qaIXA4sV9UHRGQ/0oKhEvi0iFymqnOBU4BDgWoRWZIZcomqrgRuF5Fa0t/dSuC8ib45g2Fb83Q4zMXt7SQyfSWej0T4c38/d02bNsLnAVDrdPK7+nq+0d5OUpVIxmE+QC5z1QApYFFxMU+Ew4M+ECG92J6TZTpZOYo5ajRxMKBJjKVZDHB8lvBrzlEI0AZ2c7t5NkfrVwWmOJ2cVlbG2xnfzQmlpYPf2SllZdzR3z8ijNcGelMp/i8cHtJjY1OZ6/Gwj9fLy9HooGPeSVoQHJunflY2kuk9/v86OohlTHcDZUgK6bS4vVFQFouqPgI8MmzbD7Pev0TaVDX8vL8Cf80z5pHjmqnBMMlIqfLdjo4hTXwGOszd3Nc3uGAMd1wfWFTEUzNn8lo0WrBzF9KL5efLyji4qIg/9ffTm0qxwOvl4qqqIU/cs1yunHWfBsaYKP4ZDPL16moUuN3vH+HHUODBYHBEOZEBpjqdLMnjI/h2dTXdySQPh0aWsU+osmGMZMDxcF19PTf09nJvIEBclY8XF3NRVRVFBYbGHl5czC0NDdzQ18f6RIJ5Hg/nVlYyc4K0oK2JSXs0GLYQ6xOJIdFOAySAx0IhvlBezk+6uliWKWB4qM/HD2pqmOJ04gBWRKPjWsAdQL3LxfyiIk4YxZl6ank5f85kd28KY2kWA/TYNgvXrmWB15u3C2BXKsVHfT6eHeas94rwP2M4lA/1+VgWDo/QupwifGQCa0K5RbiwqmqwI+GmMM/r5Xf19RM2p23F5MweMRgmAcWjJO/5HA7OaGlhWThMkrQ56alwmM81NxNX5Vc9Pfyxb3zpSi4Rzm5p4ZSmJn7T3U13nqfsBqeT3beS/TwOPBeN5v0edvV4+HldHR/z+XCT9g14RZhiWVzR1cVlnZ205rmPTxQXU2lZQ5563aQ1qANMmfItghEYBsMWYorTye5u9whHcpEIC7xeelOpIeGaKSBg2zwUCHCH3z/ElFUIsYwp5s14nBv7+zl0wwZuz6NJjDdXucbhYHMMKA5GmjO8Inyjqopih4Nr6+t5fOZMzigrQ1VZn0yyIZnkH4EAJzY10ZZDaHgcDu6cNo1PlZRQIkK5w8HJZWXcOnXqpEuImywYgWEwbEGunjKF6S4XPhGKRXBnmg1VW1ZOgRBW5fVYbMJsxT/r7mZ1jqioY8fR+9sB/KmhYcxFOJdQGCCV2S9Zx55VVjYkeqtIhNv9/iEhxUnSuRp/yqNtVVkWV9bV8cLs2Tw7axbframZtGU3JgPGh2EwbEGmOJ081NjIq7EYHckke3m9NDidPB4K4REZYX/3ibCX18s/g8Gc4w1EPcVUcZB2HI9WkEOBG3t7+WldHf8OBnkoGMQjwseLiynLjDOWHrPA42Fnj4evVVZyTU/PiByRAZyk80ySeTSjBB9EYdnALf39rIxG6UilmOfxcGRxcc6EvCTphMYBIrbNUxnfxUFFRdSbCrRbDdFxqr3bkoULF+ry5cu39TQMhs0mqcpnNm6kOSuL2Em60N/D06fz064u7gsEcpbb8Irw1YoKDi8u5sa+Pu7PI1wGqHA4mOZ08nY8PsSJ7qCwqCgH0Oh08v2aGioti3v8fl6LRlmVSExoVNXAsp/rng/yerlp6lSWRyJ8pa0NSM89BXy5omJIeRXDSERkhaou3NxxjO5mMGxBVJVwdAPd/c/jD72NrWl9wCnC7dOmcUxJCS4+MNU4gdv6+vhPKJT3yX+gxeost5vTysrGNBP02TZvDhMWUHgIrQ1sSCY5t62Nyzs7uaiqik+Vlo7bDzIWSdLVcYf7Srwi/E9lJTHb5oK2NkKqhDI5KnFVbuzr45VodIJnY8iFERgGwxbC1iRrW/7M2pbbaOv+D80d97N6w2+IJ3oBqLQsvlRRgTPTVztJemG+ureXvmHVWIcz0FN7L6+Xb1RV5a2tNNG8GY/zzY4Oaixrkwv6jUYE+GhR0WDEVLEIl1ZXc2BREQ8EgznNXTFV/hEITPhcDCMxxj+DYQvR0vkwkdj6wc+2xrFTcTa238NOjecA8LuenhHO77GMxF4RTsvKszirooJPlZZyU28v9/v95O6tNzHYpLPVv19djVMEJtikbQHXNTTQnUrRm0oxw+Uiatuc1dLCq9FozmRDhYJqURk2H6NhGAxbgFi8m/7gypz7ovFWkql0hvIbOZoB5UMAjwjnV1Sw/7DihdWWxbeqq/l6TQ25yxpOHAr8qa+PWxoamO50UiRCkQhVDseoXfEgHZ77EZcrr0Y00C+82rLY2e3GLcJ3OzvzCgtIR1d9soAyHYbNx2gYBsMWoLv/+VH2KpFYK6W+nZnhctFeQOMhJ3CYz8dltbVUWhYpVVbH47hFmO1ysTaR4KL2dlqTyZyVbic6tOU/oRBX1NXxr+nTWZdIkAJ2crlYk0jwm56ewez1bIpEOKOsjOPKyvhLfz+39fcPMbv5RPh1pgHVAP5MXah8wsIN1FgWL4XD1FjWqC1kDZuPERgGwxYglhjZvjWbeKIL2JnzKit5va1tiFnKTbqulEXawe0VYZbLxVV1dfgcDp4Nh/lWppidDUyxLPptm37bztm69ACvlyN8Pu72+3lvHDWWGhwOWvOYegYK8YnIkDLiO7ndXFtfz8VtbTwdiQzel5N0b4g/9vXxx/5+ZjmdfLu6moeCQXpSKT7u8/HlysoR7U9DqlgieetNxYGNySS3BQLcFQzyubIyvjkJi/pNFozAMBi2AD7vDMLR9eR+trdwWmkTyoFFRVxRXclVPd34bbDEwUmlpXy1spKnIhGak0n2cLs5qKgIhwitySQXtbcPqVG1Po8QcAP7eb28GI3yYjSKAHu53bQlk3SOYfN3AG15jhEYYRIbzi+nTOH2/n7+7vcTVqU30241BaDKu4kE1/f28tjMmTm7+g0wJdPRLzqGFqakhesdfj+fLilh10lYOnwyYASGwbAFqC7fn17/S6TskaW7LYebUt+uAARCq5jVfS9/VCGIB68mmO5ZRKD3JXaJvM/uVik17oNxyM4A/MPvz5sYN5w46QZI2YzW1zubfOJESDc8+k5Wpd0k6YUkOxPcKcJZFRWcVVHBbX19XN3bO2TeAwv8E6EQR4/if3CIcFltLRe3teVtwJRNQpVloZARGFsIIzAMhs3EH3qHzr6nSSYD+LzTqas8Ao+7hjnTzqWl60FCkTWZIwXLUcqsqafjcLhIJINs7LgH1bSGUJxZEls6/8lAWl080c3G9mZqKw7B1iR7+FfzRYp5mN1pYvRqrlsCB/BQYyM1Tid3+/1c29NDt21T6nDw0aIiTi0rY3+vd4jwaE0mB01Y2URVubK7m8u7utg3U4Z9To4qs4f5fPyyro6LOzrGzB1xAC5TGmSLYTK9DYbNoLv/Rdp7HkN1wC0rOMTFnGnn4nGnn8KjsQ42tP+dRLIPEFzOcmZMOZlQdH3m3PH1bkghJHDwSw7nNaZmrppeLAcMN1vC0Q3psNfX5szhH34/V3Z3jwgJdgMf8Xi4uaFhsKbTo8Eg3+vsHLX5k5B2et/d2Ji3T8SXWlt5ORrNWyod0lFkDzY2Mm0S9prYkphMb4NhG2Nrko6e/2YJCwDF1gSdvU+kj7ETrGu9lUSyh4FiFolkD+8330giGUB17Aip4VgoXlKcx3OADgqLgW5uLraMsACYm9EAftfbm7N4Yhx4Nx7n2p6ewW1HFhczw+XK21YV0vONqHJDb2/eY66ZMoVji4txk77PBsvCRVrQFIngEeF71dVGWGxBjEnKYNhEEol+cmvoSii6DiBdDsTO3dvOH3qH9DP7pnWHKydGtSi9KkPCU8d6CtxU7cNJ2pkN6d7Z+Yir8lAwyKU1NenzRPjL1Knc2tfHQ8EgKaAzmRzhk7Bh1BIfxQ4HV9bVcVltLUlVvA4HfakUT2Q1oKrO0fbWMHEUpGGIyDEiskpE3hORS3PsP1REXhaRpIicNGzfWSKyOvM6K2v7AhF5PTPmNWIK2BsmGU6rGM2z2CdTQTa2300s0ZX3mESyG6dj0xPOXKT4mL6NDrPsJ2HUUiG7ulzs5/VSa1ns6/HkTbZzAdMsi2mWxXElJTw5cyYeEX7d3T3mk+ZwcVLscHBBVRX/mjGDv0+bBnn+uzcWoB04RfBmzF0VlsXxpaWcUFpqhMVWYEwNQ0Qs4PfAJ4Am4CUReUBV38o6bAOwBPjmsHOrgB8BC0k/1KzInNsL/AE4F3iedL/wY4B/be4NGQxbC4djoGxg7ud1f+gtxnomS9p9iLhRLSx6aTgn8SrT6eV3fGzI9tH0ln5V7p06dfDzm7EYP+rs5O2sCKoiEea4XPxl6tTBxbkzmeSzTU0EbDtvIh2kF5Wji4vz7q+0LI7y+Xg8FBqSZOgV4ctjtGU1bFsKMUntD7ynqmsARORO4DhgUGCo6rrMvuFBDIuA/6hqT2b/f4BjROQJoExVn8ts/wtwPEZgGCaAdIXY9cQTPXg99RR5po590iaQSkUQHCOe8Icydo0jt7OaWKK9oGOH4yHFgWzgToJ0kdZWnMARPh9PRSIjopOED/wQA8z1eLinsZGWRIJ7AwE6kkkO9vk4qrh4SIHBG/v68Nt2TkE0IDZ9IlRbFl8f1v96bTzOc5EIJQ4HRxUXc2VtLVeI8HAohAClDgffra5mwRj5HYZtSyECYxqwMetzE3BAgePnOnda5tWUY7vBsFkkUyHWtfyZRLKfgdZARZ5GZtZ/LqMRTByWVZTX3JSNYKGj1J6NJVrHHEHEiUNcpOzwiL0pHHyEProowSdChWXxvZoa9ggE+GNf3xChoUB7MklcdUTC3FSXi68OW+iz+b9M//HhFAFnV1QQVmUPj4dPFBcPjq2qXNXdzd2ZarIWcEVXF3+or+cndXV8z7YJ2jbVloXDWKW3ewrxYeT6FcdTLy3XuQWPKSLnishyEVne2dlZ4GUNH1aaOx8klujG1jiqCVQTRKIb6eh9csKvFY0X9veoQJFn5iZeRWio/hSCkLJzO4R9AidWNnJOeTmX1dby8PTp1DqdnFNRQX0Ou/67iQQ35Wl5Ohr5fAQp4JSyMi6prmZxSckQQfRMJMK9gQAxVWKqhDOvC9vbiatS5HBQ63QaYTFJKERgNAHTsz43Ai0Fjp/v3KbM+zHHVNUbVHWhqi6sra0t8LKGDyO2JgmGVzPctKMk6Qu8MuHXC0XWj31QBls3vcFPa/cj2Bont8nKwuuewjGVc7i4uppjsxZsv23TnKNsyKb2jzi7ooKiYQu7C1hQVERtnjap/wgEhpQxGcBWZUVkZBa8YfumEIHxErCLiMwWETdwGvBAgeMvBY4WkUoRqQSOBpaqaisQEJEDM9FRXwD+uQnzNxg+YJSErtFMQgDReDvtPY/T3vNfIrGxTERpHI6xonIEsLAcbmLxjoLGHMloXbsdlPp2YWb9GekjVQlF1tHR+wQ9/S8RT0XzRksVWl4kmyOLizm/ogKvCCWZvIe9vV5+WVeX95x8RQNDqvywq4tHx2gva9i+GNOHoapJEbmQ9OJvATer6psicjmwXFUfEJH9gPuASuDTInKZqs5V1R4RuYK00AG4fMABDpwP3EraBPovjMPbsJk4HC68nnqiseHKqgzWbspFZ+//0dn35GASXXf/C1SVHUB99VGjXq+8eC9aux7Ouc9lVVBWMhe3s5rW7ofYMql0NuHoBvpDb1FZOp8NbXcQiq5HNYGIE+ExZjhP5f1hSoYLWDRKFNNofKmyktPKy1kdj1NjWWOGwX6qpITnIpGcWkZLMsl3Ojvpt21OzmoIZdh+MaVBDDsU0XgHa1tuRjWFahIRF5bDy5xp5+Bylo44Pp7o4b2mP+Qsz+F0VlBRvCfVFQfitHIvsK1d/6bH/8KQbSJO5kz9Em5XDas2/Bo7RwHCiUTERXnxnvSH3hiWdQ4bpYEf6GEkEeI48ZKgkii3N9QwpWjLx5nYqlzc3s6zeYQGQLnDwdMzZ2IZP8YWY6JKgxiBYdjhSKbC9AVeIRbvpMjTSHnpPCxH7vS07v7nae9+bAyTlZO6qsMJR9ahmqKidB7lJfMQcaCq9PpfprPvKVJ2GK+7gfrqRfi802jtWkqPf7RGShNH/kgsiyBunmAm7ZSyL03sRRteq5SPzLiYrZEvq6q8GI1yXmtrzoqzHhH+M2OGSbzbghiBYTBMAOnigf8ZZwFAwWmVMqP+dIo8QzvEqaYIRzegarOh7W50RP+77Qe3q4baikMpK94VRx6BOpGc2NTEOznKqxeJ8OysWaP2xTBsHqb4oMEwBrF4J8HIGpKpkbkLA5QV774JIyvJlJ81zTfSH/yg4EEouoFV63/Fhra/s7F96woLp1WKSC5/Qv7/4vFEF82d97Nq/a8IRdZtsbkNcEFlJd5hQsGbadtqhMXkwBQfNOxwJFNhNrT9jWi8A8GBrUmclo+UHcESH17PFOKJ7kxeg+CyKogPVpMdDylauh6gtHhXVJNsaL09E/46ERReIlDESUPNYvoCKwlG3s/4bpwIQnnpfHr9K8gfaWVja5z1bX9j1xn/D8vaco2Hjiwu5vvV1fyqp4eQbWNlhMVFoyQLGrYvjMAw7HA0td+TCY39oMd1MpXOO0iqn2DEP+T4lB1iU5VttW16+5eTSPkHM8snhnxjOako2ZNIrJl4sg+Pq5opVR+nxLcTpb6PEIk1E4quw+nwUVayBw5xEYt3Eo6uHf1qmuDdDb9mev1plBTNnsD7GMoJZWUcV1qK37YpdjiGlB4xbP8YgWGYVAz43PI5a5PJIOHYBsavLYy/jhOAkqC997+g9pi5HpuGkC4NknYI15QfRG3l4TnvX0TweRvxeRuHbJ/VcCbvNf2eeKJ71CvZGmdD2x3sMv0iXM5Nr6I7Fo5M+RLD5MMIDMOkIJkM0tL1CIHwKgBKfDsztWYxLufQ+P1YohvVTVv8cyE4qS4/iG7/s3mbHQ0PZZ1IBAe1lUdQXrI7Tqt0k+phiQiNdSeyruVWbE0yqnBUm77gq9RWHLzpkzbssBint2G7R9VmTcvNGWFhAzbB8GrWNN+EbX8Q3RRP9rOx/e9MZJKcw+HC7aoaLYk8LyIuXFY56XzXMY7Fyn2cCOUlc3G7qjareGKRp4Gdp19AdflHsRwl5OuYoaRIJsdfNsTw4cAIDMN2TyD8LqlUiKFPxkrKjmV6TqTp6FmWt0DfpiHYdpzW7kfI7zQeicuqxOHwopogkepHxIHPMwvL4ct/JbEoK9kj46y20i9xUl91NG7XxPSIcDnLqK8+il1nfoOG6mPJJTRE3JT45kzI9Qw7HsYkZdjuiSW6sHOYfVTjxBJdg59DkTXk1i4Ep1VCMhXMPMlrpofFWGqDpv0S4+y7nUj1DRlbNUE4tp45086h1/8yvYHcuUQVJfOYUnkkgfA7AJQW747bWT6uaxeCiFBVvpBwdD3+8KpBk5qIiyJ3PSVFO0/4NQ07BkZgGLYaqQQEWsBXA+5xlDLyuupwiCtHyKoDp/WBc9ayfIPRUEMRdm78CqrQ1HEXoehGBsw/6WztMez6ozBr6V0ArFt0StbW3H2+Wzofobbi4JwCQxwuSormIOKguvzATZrLeJlW91lKgq/R438ZJUVFyTwqy/ZFxBgeDLkxAsOwVXjxWnj8B2AnQW2YfzYc8xuwCjDLl/h2xuksJZ7oYehibNPR+zilvp1xu6qoKT+Ilq6HczihhUB4FbFEN+HoRrKbl27NQgexeDut3bmLFdZXfWKrL9QiQkXp3lSU7r1Vr2uYvJhHCcMW58274bFLIdYPiRAkI7DyFvhPVgf4ZBReuBb+dDDcdjS8/Y8PFnMRB7Mazh4MLc3GtuO09zwOQHnJPKrLD2CkbT5FS9fD9PhfKqhDXiHMWnoXs5beRXF7E8XtTYOfR8PhcGHbuRP7AuH3Rz1XVYnGOwhHN2DbWy4qy2AYDaNhGLY4//0OJIZV50hGYMVN8PGfgzjglkOh880Pjtv4LOzzJfjkbwfOyGcy0sGyFiJCTcXBdPc9NyInYqD73rak2DeHYGh1zn32KM76eKKX9W1/I5HsR3AASn3NsVQazcCwlTECw7BFifmhN8/Ds6bSWsfaZdD51lChkgjByzfAQV+HilngEHde+5FlFQ2+TyZDiFh5cyYmigGfRVqrENYfc3q6D8Uo/bv9wXeQHP4NERflJXNznqOqrGu9jUQy7UgfOLu162G87lqKPFMn4G4MhsIwJinDZtO3Hp76CTz6/2DNY0PX9bfuTWsQubBcaQf4e4+kBcRwxAnrn0r3zl7TclOehdhBVen+BMPvE4qsx5mj58XmIeTLWRBxIuLE666nsnQBJUW7UF66LyL5nsOSuF21mf2SGcOF111PWcmeOc+IxJoyIcVDBY1qgvbux0kmTcc6w9bDaBiGzeKte+G+M8FOgR2HFTfArCPg1PvAYUGoPX/w6uxPpIVJcT04XDDcNJ+KwXNX2yz/exepyFEUz+hjzhnLKdspu8SFTVvPv3GIC1BEXFSW7UuPf8Vmm6C87qnUVR7Kxo57c46latO6+AJiyU4k8AqKjdMqZbTnsESqj9lTv0ivfwVJO0xZ8e6UF++R0z8DkEpFyCewQtE1vLvxN1SUzKehZvFW6W1h+HBjNAzDJpMIw/1npf0RA77ceBDWPg5v35v+7Cknr/sh0AzBdtj3f8CR49HFTkD7SgdNj+xG67LdeP+v+/Hfz3yZlsd3GX4ktsawNU7KDtHd/wLlJfM28+4sKsv2obR4V2bWn4HbWcXIhdsmmmhFNYmtsXSiXrIHHaVireUoosjTwNTaT9FY+1nKi+fmFRYARd5po/TqUFRT9AVfo7v/hTzHGAwTR0ECQ0SOEZFVIvKeiFyaY79HRP6e2f+CiMzKbD9DRFZmvWwRmZ/Z90RmzIF9+TvJG7ZL1j+d1iKGkwjBa3+FWACevjL/+W0vw62HQdVO8NnbwfLmOzK9UGvKIhV1sfxbx6Gp0Z6mlb7AilFMQ4VgE4m1YNtxiotmsvP0C7Ecm9YH+wOcVJcfSDTWxvvNN/L2up/y9tqfsrH9XlKp3E5vp1VMTcUheXpdpFFNbLXOftkEwu+xpvlPrFr/K9a33kEk1rbV52DYuowpMCT9+PN74JPAHsDnRGSPYYd9CehV1Z2Bq4H/BVDV21V1vqrOB84E1qnqyqzzzhjYr6odE3A/hq3IaDkU4S745ZS0FpEPOwk9q+H3e0DL8nyGl5Ek/F7+deSFvHDxCfS/W5v3OFUddaEdHaU/+HqmXlUCEcHWzSs7Yjk8xBO9rGm+iWishYFM8kDobda13Ua+7pd1VYcxfcopFHvzlx1P2Vu3s19f4HU2tt9FJNZEMhUkGHmXtS03Z8rKG3ZUCtEw9gfeU9U1mta17wSOG3bMccCfM+/vAY6SkQbVzwF3bM5kDdsXMw6BXNYUlw9aVqRNVWOhNnSvgmd+ns7FKARNOQg3VbHxwb147NPn0v7srDxHpijx7YLbVYPD4cXnmYnX3VDYRQDVJPFkH72BVwAo8kwr+Nycs7FD9PifHxnyS4pYvJNoPP9iW+rbmZkNZ2ZMY8MRir2zNmtu40FVaeteOsKvo5qgveexrTYPw9anEIExDdiY9bkpsy3nMZo2uPYD1cOOOZWRAuOWjDnqBzkEDAAicq6ILBeRJtx76AAAIABJREFU5Z2dnQVM17C1sFxw2j9heJO2kmn5I6PyMa6W2oO6iKBJJ8986XQSwZE9qZetruKkG50c/ftZnP23PfnXW1Gi8fGZTVQTgyXV66sXkTtOZPNdgYKM2a9CRJha+6mM1jTwHVg4HB6mVH9is+dQKCk7nDdvJBJr2WrzMGx9CvlLz7WQD9edRz1GRA4Awqr6Rtb+M1R1L+CQzOvMXBdX1RtUdaGqLqytzW9+MGwb2laOFA59a9IRTlsLO+5kzR0LhmxbtrqKa5+eSUfQjQJtfvjtk1NZtrpy3OM7M74LEUcmEukDgTVRgYaKjcf9/9s78/CoqrOB/84s2RMSIAFCgARB9kUIiIosIorIogVFRAFB0Sqotbaf1FbrrtW6tNoqrbsIWNzYhLZSF5BdEBBEdgj7GkLWWc73x7mTTGbLTGYySeD8nmeeuffcc8+8czO57z3n3ZpU2S8xPofWze8gNakb8bFZNGrQmzZZvyTWGr0ypyZTnN8nAqs50m7NmrpEMAojD2jhtp8FeD5GlPcRytLYADjpdvwmPGYXUsoDxnsB8CFq6UtTz/j2Se+lJ+mgeiUp/Ic8VHnioaUXVmp5d01zSu2V18tK7WbeXRPqspIgLaUHAAePLTCWYVxfTqLyUoVfsElKOweOfo7NfqbKvnEx6TTPGEnr5pNp2ugqryJSNY1JmElL6ellHxLCSnpav6jKookuwSiMNUBbIUSOECIGdfOf59FnHjDB2B4NLJWGBU+ojGo3oGwfGG0WIURjY9sKDAM2o6lXSCcURtBVwRzr2yYShCTENVYBbC678XEfS1SB2gNx8swapHRSXJpXHeGCpqTsEHsO+Td+1yWaNhxMWnJ3I3jRiknEktFwEA38BCBqzg2qnE9LKe1CiKnAElRO6LeklD8KIR4H1kop5wFvAu8LIXagZhY3uQ3RD8iTUu5ya4sFlhjKwgz8F/hHRL6RJqJIJ3z/pso2W1oA7UdCv9+rCG1hgpQsOBOh+2junbD+LSirRsG3NhNWYS+2cGR5azKv+JnGSWUcOxvr1a9xkv8YCd9IzhRu4ae9e1HTn3Bu5kZ9boSR5MNzZiKx2c9QXHqQhLjwDOw1jRAmmjUeSpOGg7E7i7CakwLGk2jODUR9eJpxkZubK9eu9V18RlMzzJ8Cm2ZW5HkyxUBSE7h7M8SmwMpXYMn9kfksX9HewRDfNJ+LHl/Itjf6cmJdC4Z+8wqrisz89dtWlZalYi0Opl2+l4FtTwYYrSYx0arZOErLjnK64AefBniTiKV5xvWkJLarBfk05ypCiHVSytxwx9GR3hq/5O+Dje9XTgroLIOiE7D+bbXfeypYw41nc41dzUwexYcb8N2UmzmxriUg+PHlAQxoc5Jpl+8lPbEMgSQ9qTSAsgjVcGLCJGKMPFLBJ/8zCQsORxGNGvShQVI3nzEiTmmjoPBnjp78yqj/odHUHXQuKY1fDq4Fc4x3fIS9SCUZ7HOfSkl+0SRY8zePSqYCGrSC/D01J5+w+HbHPfCfjnQ8+BWWp65h+Hc5gKBRz33kXhaZtTOrJYWmDQcTG5OOxZLMT3ueJxjDt0TlnwJIS7mIk2dWYbcXuMVlqCWv02e/B0wcz19OZuMRpCZ3iYjcGk24aIWhAcBeCjsWQ8lpyLkCGrSA5ObKhuGJyQKpreCDq2HfMmXLMMeo9pgklY68/6Ow6J6aldlf7Ia0xfDNTVMoPBILDrUkdXxtS5aOnszQr1/BklAxlRGYiY9tTnHZwQA5myoTa00nJUklO3A6bYZNIgh5pQOrNQ2HoxizOZ4Lmk/heP5KCgq3InFSZjsN5crDiZRODh6fR3LihZhN3vYYTXSw2Wzk5eVRUhJepH80iIuLIysrC6u1uhkOAqMVhoaD6+CDq1TNbSnVjbjPA3DFk2qWcHxb5ZuzOUal9dj7TeXZh8kKF1wFN85V+6f3Rvd7uLAmgK0gHtzzTTlNOEos7F/YiZwbNgAmhDCRENuCFk1uJP/sJg6f/DcCE1I6/VbmE8JK49TL3PYtxMe1pKhkD5UN4i4fYScYRY8EsO/QB0jpICWxI5npI2jScCBNGg5k/5G5PgP3BGYKi3eTktg+3MuiqSZ5eXkkJyeTnZ1dpzMCSyk5ceIEeXl55OT4TyMTDlphnOc4HfDhUCj2WC5f9Qq0uhz6PaxqcefvUzMJS5yqhPfDu95LVU4b/PQ5fDwO9n0TavS2B2E4JJWc9O2e6yiKpWB3I8CkMtBaGxJjTQWgYYNepCR1orB4N0dP/o8yu++o6yYNryIxvhUAhcV7yDv6sZHHSRpiW0FIkuLb0LjBpRQUb6fMdoqCop+Q0l5e2OlM4VaEMNM8Q2XZCeRhpL2PapeSkpI6ryxAZQJo1KgRNZkRQyuMc4SzR8BRptxcQ/ld7/8ObD5yPtkK4aPRSkk4jVg1R5magax7w3fBI1BKYvMswvM+BdpcA0d+CJy8MBC+ltIsiaWkdTxMWvJFJCW09j5H2iks2etXWQDlQXI2+xn2Hv7QK5+SECZyMicRF6uSLyfEt2BH3utey10SO/mFm2nmvAaTKYa05O6cKdzqs+5GYnzNPC1qgqeuKwsXNS2nVhj1nNN7Ye4YI0WHUArj+g8g6+LgzrcX+1cwtiK8bvzOsoraF36JgKf2sS3eOapCQgrMMU4cZcoRUFgcxDYqpMWQvaSn3VGpa3HpIYpK8jh68kucVRRdcsoySsqOcezU137KwErK7KfKFQYQsCqew1mCyRRDYnwODVN6c/KMqmshMCGRtGw6BlNYado1msihf4n1GKcd3r5cPYW7nqhP7oD3r4RpO1S8BChvp6+fUB5NTbtDvz9A027qWIvL1Dg+qcUQnXC9q4QV2l1vYtd/HDgdDlpcs5O+jx+hec6dWC1JgEoJvvfQB5SUHTFu/lV5Opk4mb+KkrLDhreTd3+JxO6oHHmYENeCgqKffI7nXmOjaaMraZjSk7PFOzCJWJIT22ljt6ZOoRVGPWbnv5VXk+fyS9lZeLE5tOoHXW+BL6YZy04STu2CHV/Arf+BFpdCTCIMnwGfTgzT5lDHsMbDRRPhhtlmVDKBDsargkPHFlJSeshPrXBvTMJqKItAF0qSENeyUkuThoM4W7QTicfsRTo4fnoZGQ37lzfFWNNoaO0VlDyauo3dbsdiObdusTpwrx6Tv98j9sEN6YA9/4N5t3ssLUm1v/i+ir6mGP/j1FecZdA8QDpLKZ2cKdwStLIAjDKs/pWFEFZSEtoTF1O5eGRsTGNSEjt49Zc4OJ6/LOrFjzSR4YknnqB9+/YMHjyYsWPH8sILLzBgwAB+97vf0b9/f1555RX27t3LoEGD6Nq1K4MGDWLfvn0ATJw4kblz55aPlZSkZr1fffUV/fr14/rrr6djx47cddddOJ1OHA4HEydOpHPnznTp0oWXXnqpVr7zuaX+zjOCslP4WVY6uBb+0gYatoWdiyMqVq1jskLf30F8gIzfEicyAllmXcTFZNIwpSepyRf5PF5i812HQ2CmtOx4nc8dpanM2rVr+fjjj1m/fj12u50ePXrQs6dKsX/69Gm+/vprAIYPH8748eOZMGECb731Fvfeey+fffZZwLFXr17Nli1baNWqFUOGDOGTTz4hJyeHAwcOsHnz5vLPqA30DKMe07Q75AxScQfV4dTOqpVFg1YQ26B649cWrqSIgTAJS0jV94yRfbYmxGVzQdYdpKX08OulEmPxXYdD4sBq0TUk6hvLli1j5MiRxMfHk5yczPDhw8uPjRkzpnx7xYoV3HzzzQDceuutLFu2rMqxe/fuTevWrTGbzYwdO5Zly5bRunVrdu3axbRp01i8eDEpKdFNae9CK4x6zo0fw4DHoYE/z0uTSqFRXS4cDnGp1T8fYMKEa5kw4drwBgkBRyl8+3TV/TLTh6ucUKg4ByEsmEQc6an9ER4XTQgrqUldMZliK/ob1e6aNR5a5Wc1Tr3Mu34EZhLjsqNez0ITPoGStiYm+k+u5nqgsFgsOJ3O8rHKysq8+rjvp6Wl8cMPPzBgwABee+01br/99nDErzZaYdRzzFa49Ndw/7pTtBvm8HJFtcZDl7HVH3/Nq1BadU2fOkdhEJVY42Ob0qbFVBqlXkZyQjvSU/uR7riXpZMHsOqB6yg5loR0mBDCSsOU3mSmj6BNluqflNCWRqmX0SZrKnExVVeCTIhrQfP0kZhNCQhhRWAmKaEtWU1GR+DbaqJN3759mT9/PiUlJZw9e5aFCxf67HfppZcye7YqBTRz5kz69u0LQHZ2NuvWrQPg888/x2arcIhYvXo1u3fvxul0MmfOHPr27cvx48dxOp2MGjWKJ554gu+//76Gv6FvtA2jriIlfPcdfPIJxMbCzTdDZx/Fab79Fm6/Hfbsoat9JLtN7+AgHhAkZcKYTyGrN5zYBgdWV0+UklPVO881q8jOXlZp/913ff9zRZLMIB2NrJZkmjQcCKho91cvUe/S2Yl9n3UkrlEZzXpaGb/EZPRPKu8fKg2SOpGS2AGbPR+zKR6zOa5a42hqn169ejFixAi6detGq1atyM3NpUED77Xbv/zlL0yaNInnn3+e9PR03n5bpXm+4447GDlyJL1792bQoEGVZiWXXHIJDz30EJs2bSo3gG/atInbbrutfFbyzDPPROeLeqAVRl2gqAg++ACWLoXWrWHKFHjmGdVWXAwmE7z8Mjz9NNxvFJ/4+WeYOBFWrABgD/34lLexOysMGqWnYctHSmFc+Ry8ewW1GlsRFYSaVQ3+U+infv8mlBW6uykLSk7EkrcMjmyEJl0jIJ4wEWMNva64pu7x4IMP8sc//pGioiL69evHr3/9a+64o3JQaHZ2NkuXLvU6t0mTJqxcubJ8310BJCQkMGfOnEr9u3XrVmuzCne0wqhtTp2CXr3g8GEoLISYGHC5zLmyYzocSnFMnw5jxqg+ffqocw2+4jHsVF47tRWptOMdb4R/P0iNK4uUFiqBoWs5yDWTiNbMIrEpJKar2UXRCXXjFyEsuh5c412fHFReqqObI6MwNOcOU6ZMYcuWLZSUlDBhwgR69OhR2yLVOFph1DbPPAP794PL6FUWIO+G2QyLFilF4ZFq+QQX+jxFCMl7gwQ2/9kpIkbBwdqL54hLU0kHS0/D0U1qZtWsB9zyb7AEGSyd0QW2zQeHR1JF6YRGvi9v5X7SSUHRzxQU/YzZlEBacndiYxqH/mU09YIPP/ww4mMOGDCAAQMGRHzcSBGUwhBCDAFeQYXM/lNK+azH8VjgPaAncAIYI6XcI4TIBrYC24yuK6WUdxnn9ATeAeKBRcB9sj7Vi40UH38cWEm4I4SaXWzYoGYcbjRhI2dpiqcfg7PMgcMenecCf8rCNbMwx0BGV2jQUiUWbNgWzuxX6UwcYcauedpZys7CgbUqUeLF91Y+5rTDvuXqM1v2VW7JpWeUncdTWZhjIaMzNOsZ+POldLL30AcUleYZCQRNnDyzShdA0pxTVDlhFyq38mvANUBHYKwQoqNHt8nAKSllG+Al4Dm3YzullN2N111u7X8HpgBtjdeQ6n+NekwAFzwvnE4YPhx69vTKGDiQP2ClshKxcpb0uB2RkDJsOo2Bu7eC2QI7l8DZQ7B/GSDUsWrZf6tIzGkvUmnY3clbBX9uBrNHwL9ugOczYPMcZd/Z8i/vMdpfD7cs8Z+gscx2iqMnv2bf4VkUluxzyzbrREo7B4/PxxkgW6PTWUaZPb+8Ep9GU5cJZoW3N7BDSrlLSlkGzAZGevQZCbj+NecCg0SAPLtCiGZAipRyhTGreA+4LmTpzwXuvhsSgoi8i4+HDz+E1FRl7PaYjDVnLbdwFc1ZiYViGrCHq/kVnPWThzzK7FwC7/RXT/22QmVfKTurnup/nA2m6pR8CGI+6l5KwlakqgQWHVczitIzSpbPxsPxrSp9uzvWBJWPK85P4OLpgk3syPsbx05/w9niHeAjzYjARFHJPq92p3Rw4Nh8ftr7PDv2v8ZPe5/n5JnaN2pqNIEIZq2iObDfbT8P8ExKUd5HSmkXQuQDjYxjOUKI9cAZ4PdSym+N/u4FlvOMNi+EEFNQMxFatmzpq0v9ZsoU5T77r38pJVDqZ22mQwfo3Rs2boTf/tZnl5Z8x+1cUqltI7dGWuJqUXJavTxxGg/knjfrSCAsavYCUFoASx/2LvoEaonK1wO+rUiljfeFw1nKwePzgijrKr2CAAEOHV9E/tmN5edLaePwicVYzUkkJwZhMNFoaoFgZhi+Zgqez3b++hwCWkopLwIeAD4UQqQEOaZqlHKGlDJXSpmbnl51gFS9w2SC995TimBogIjh9etVHEafPrBkSdDD5/J3rETB4u2GqWbKCYeMdMDqV2H3UngxE9bN8G0rkU58/iKtCf49owqL95RHfAdCCItX9lqHs6ySsiiXQ9o4dvqbKsfUaGqLYGYYeUALt/0s4KCfPnlCPU41AE4ay02lAFLKdUKIncCFRn/3bD++xjy/aNsWfvMb+Pe/lXutJ1LCyZPe7VXQmdkcpitOYpCY+YnryCcbUDYOG4lUaQwIkWh6SsU3Uq6wUvpwiZVQdAxmjfBfIRDAmgiJGaquiGumI0xKYXTzM0ETAf11BSZhBWGiVdObvfo6HEX4u+Y2e36AcTWa2iWYGcYaoK0QIkcIEQPcBMzz6DMPmGBsjwaWSimlECLdMJojhGiNMm7vklIeAgqEEH0MW8d44PMIfJ/6TZ8+MGKE8oSKEAIYzO+4it9wJQ9xDx0Yw7X8HgvTSWY0ozFTQiSDNIK135qsYLJQbX1lssC4RTBpuTJO+8qZZSusWPbyhTURWl4GU9YaxvdYZfdoPRhuXwWxftI8+SubKoSFRg0uoXnGdbRr+WvifWShtVqS/Sqc+NgqsiZq6jybZsLL2fCYSb1vmhmZcRcvXky7du1o06YNzz77bNUn1ABVKgyp5s1TgSUoF9mPpJQ/CiEeF0KMMLq9CTQSQuxALT09ZLT3AzYKIX5AGcPvklK6HpN/CfwT2AHsBL6I0Heqvwihort//3uIcOEVgcRCKVZKaMcizDgQQCc+4UHSacCeiH5eEALRdzpMWQdJTcESH+LpZvjFTFXzoml36HSDivD2xBJX2fDtTmIGDHsDbl6oUqFf/x48XAx/sMEtiyHNu+x3OSZhoWXTMQhhVQkMhQUhLDRM6U3TRoNJSeyAyeT7byiEmSZpg7yTEQorGdVMO6KpG2yaCfOnQP5eQKr3+VPCVxoOh4N77rmHL774gi1btjBr1iy2bNkSEZlDQdSn0Ifc3Fy5du3a2haj5nE6oUsX2L4dbIFrTEcCCWxhFHOZW2XfoBBUOWFp0g1unAsN26hloO/fVJUBg1nO6norDHkF4t0ybDhs8NcL4UxeReVAYVZLVrazRhEpN6yJcN070DGI3H/2Uti+EM4egVaXq7iM8s91llJQuA2ns5SkhAuIsQYowuHBmcKfOHb6W+z2M8THZpHRcKBX8SVN7bN161Y6dPAugOWLl7MNZeFBg1Zw/57qy7BixQr++Mc/ssSwX7pSiUyfPt2rry95hRDrpJS51ZdAoSO9o4GUsHo1rFoFzZvDsGEqoaA/TCb4+mu480747DOlQGoQAbRlUcTGazsM9nypPJKkUwXsOR3GUpWhSI5uhjd6wLV/g2XPwPFthqdwEMrGUVZZWYDK2jv5O1gwBXYsVmPlXKHKzx5YDZ9NVMrIUQYxSaqOSPvrq/4uRzfDuwOV0nDVPm9/PfzifWXnMJtiSU2uXs6QlMT2pCS2r9a5mrpJvrcHdcD2YDlw4AAtWlSYkrOysli1alV4g1YDrTBqmrIyGDlSZZW125V9Ij5e7V8YwH2ycWMVBX74MHTvrgzeNTjbsFLMGEaykNc4S3jr6CYTTF4BK15SaTpO7YWS45X7SIeKw3DdyEPB3z9fcjMYO99wk5VKiQCkZqv8UhvfVxHhbYcpZeI/UsiQUcLs61TchjvbPocf3ofuE3yfpzl/adDSzwwjzIgAXytBAULdagxdD6Om+etf1WyhsFDFWBQUwLFjKolgMDRtCj/8AL/8ZWhR4SEigAtZwBB+RbgG8MQM5Y469FU4vdtbWZQjq+FRJeCCqwN3MVkqlIWLtBzo/whc/RK0HlS1sgAVzHf2kHe7rVClHNFoPBn0lHcFTGuCag+HrKws9u+vCIfLy8sjMzMzvEGrgVYYNc0//+mV9wkp4aef4MCBwOcWFcFf/gI33qj6l/iIOosgJpwUkImJ8GYy/R9R71s/iXxAXkwS9J4a2TH94bD5z3ZrDzP3lebcpMs4tQzaoBUg1PvwGao9HHr16sX27dvZvXs3ZWVlzJ49mxEjRlR9YoTRS1I1jd1PJLDDASdOKJuGiz17YM4cpSgGD1ZpQ7Zvr3FF4U4zvsdCKWVUdu01U4IDK1QRrNZxTEU97YIDYPORLtyFMIc2w4hJhjvXQ0KjqvtGgozO6umwzCPu0RIPXcO8AWjOXbqMC19BeGKxWHj11Ve5+uqrcTgcTJo0iU6dOkX2Q4JAzzBqmptv9h1XYbNB376wZo3af/99lf7jkUfgySfhiitgy5aoKguAliyjMVuM2AyFwEY8p+jPY1We714aNauPcmv1RajR4NlXwH27oOEFoZ0XDiYzjJqlPKpcpW9jkqBJF+h1d/Tk0GgAhg4dys8//8zOnTt5+OGHa0UGrTBqmt/8xitRYDkFBWq56eRJ5RFVUqKM5E6nUiiO2ikucStXchH/JJZ8rBTSgU+4g1wu5UWEjwR77hSdqFiGatUfMnMrx1iY41SK84smEVLA3sHV/pVPpMnfDwfXKlfcnCtg2s/Q/1HodQ9c954KFIyWLBpNXUIvSdU0u3apGYY/D6ejR+GddyIeqFddBBDHWa5lGtcyrdIxBxYsFGMjye/5J7fDP3rBpO8gJlGlBl/1Cqx/G5DQbQL0uR8+HlsRLxGsYD8vgM43VetrBUXxKfhoNOR9Z7gC2+GKp5S8l3u7u2s05x114y51LuNwBHbJkbLOKIuqKCDTyD3lH0cpnNgOa16DnncqL6Pe0+AyjwS7F1wFu/7jHVDnD5cbbk0ydwzsWwbOsoqstksfVtX22gbIC6nRnC/oJamapmvXwO6wLVrAhAn+jeN1BCcmFvMywawj2Yth+XPwQlP4R294vjF880TllbluEyCpWYVtANRTfXonb7dEUEF/F1wV/vfwR8Eh2PetUhbu2Ipg+fM197kaTX1CK4yaxmyGfv28200mSEuDuXPBaoVGUXL9qSb76Mt2gn/MLj6lyp2WFaib7rJnYcPbFcdjEuGONXDpg9ConQqsG/5PuGsjtLlGGZoBEEqBXPZ/4Qc/BaLomH9DvK9YDI3mfKR+rIXUB1zpO0weOvi551RxJE9MJrjgApg1C/7xDzjuL7qtFhDCy1CfyFEPg7crj4cfPOz8tiL49hnD2G0QnwZXPKle7tzwkbJXbJ4D1jjoPklllK1JGrXz3W6y1uzMRqOpT+gZRrgcPgzXXadyQ8XEwLXXgisiU0p44gnf59ntsHatUih1RVm0awczZyrPLg/S+Ykc/oeFIrrxFkmBypf40SOFR4ITQ5ig3QgYNRNGvFnzygLAEgtX/7nycpjJqsqz9tUGb40G0AojPGw2uOQSWLhQKQCHQ1XD69MHli+H1q19F0NyJ5TEghMS1KsmcCU8/PxzeO01n13GMIq+PM0w7qYXf8NCZYu1wE7L/hWBe5407x1poSNLjztg7AK1JJbRRRnr79qoclRpNBq9JBUeCxaoaG13g7XDoXJFDRwYldTkESMjA/btU9+pyLfrkoVS+qOS4lzGnzhIL3ZyFSZjqSo18Tg3/iuH/cvh43FgN4YRJhWLMbgeGI9zBqqXRlNdFhQU8PKpUxy222lqsXB/WhrDkpPDHnfSpEksWLCAjIwMNm/eHAFJQ0crjHDYts33zTXSisI1q8i2VN5/N0if1GC4+GKVSj2Qt5abbcOMnZu4nqN05DDdSWUPLab0QaT/mfbXqQJE3zyh4jKa5cKAR1WqjUPrYeVLcHqPSjHee2r0Un1oNDXNgoICHj1+nBLj/+SQ3c6jxpJzuEpj4sSJTJ06lfHjx4ctZ3XRCiMcOneGhAQVsV3fWbQIFi9WXl3+MJnU9y0qKo9Cz2ALGWxRKdtvfbW8a6vL4dZ/Vz5966fw6S0VdTIOrIF1r6v8UElNa+JLaTTR5eVTp8qVhYsSKXn51KmwFUa/fv3Ys2dPWGOEi7ZhhMM116jkgdYQEyOFyrtF6rXHrl6u/Uhis6n062UB0ss6HGqG8X//B3FxSnnExanX9Olw0UV+T3U6YMGdylvKVfPbUaJSiXz7dGS/ikZTWxz2M0P3117fCEphCCGGCCG2CSF2CCEe8nE8Vggxxzi+SgiRbbQPFkKsE0JsMt6vcDvnK2PMDcar/tWmNJuVcfuGG2pbksjhcEBmpu+EiaBmFwkJKrPuCy/As8/C5s3whz8EHPbUTt9R3U6bcqHVaM4FmvrJ2uCvvb5RpcIQQpiB14BrgI7AWCFER49uk4FTUso2wEvAc0b7cWC4lLILMAF43+O8cVLK7sbraBjfo/Zo2FC5ol5+uXcMRqSpiZmFJ3Fxyvg9fbpaZvIkIUFFpzdpooo63XefiiepatjUihKnnsQHXwZbo6nT3J+WRpxHKqA4Ibg/Lc3PGfWLYO5wvYEdUspdUsoyYDYw0qPPSOBdY3suMEgIIaSU66WULof9H4E4IUSAYtb1gOJieOUVZSQeOFAF5UkJ772nbqLJyYHtAHWZuDgYP17J/+CDat/9xy+EijcZPTrkoRMzoOXl3tHU1kS45IEw5dZo6gjDkpN5rHFjmlksCKCZxcJjjRtHxEuqLhDMPKk5sN9tPw+42F8fKaVdCJEPNELNMFyMAtZLKd1rlb0thHAAHwNPSh+Fa4UQU4ApAC1k1W+tAAAQo0lEQVRb1mBuiGCw2aB/f7UE46qit2KFurlmZqqn7cxMFYvxySfelfbqOv37w0svqe2kJFV3/KabVBEnUIF9s2apWUY1GD0LPrwWjm5WisNeolKGdx4bIfk1mjrAsOTkGlEQY8eO5auvvuL48eNkZWXx2GOPMXny5Ih/TiCCURi+4nY9b+wB+wghOqGWqdyTLIyTUh4QQiSjFMatwHteg0g5A5gBkJubG16x6XD55BNV1MhdEZSWqiWcfftU7e3WreHNN+HTT2tPzupw5ZXKS8qdTp1g0yY4aEwSMzNVjMnXX0N2NrRqFdJHJDSG21fBsa2qGl/T7qpNo9FUzaxZs2pbhKCWpPKAFm77WeCVF6K8jxDCAjQAThr7WcCnwHgp5U7XCVLKA8Z7AfAhaumrbrNkSeDI7eJi2L0bNmyA3Fy1fFMfiItTy2z+yMyEpk1h2jRo2RJGjoT27VUalKoi2X2Q3gFaX6mVhUZT3whGYawB2gohcoQQMcBNwDyPPvNQRm2A0cBSKaUUQqQCC4HpUsrlrs5CCIsQorGxbQWGAbUTuhgKqalVG7aLilS09MKFyh4QVwul2QLV3/AkOxvWr4eOnn4MHrz2Grz1lqoKmJ+v3pcuVXXHNRrNeUGVCkNKaQemAkuArcBHUsofhRCPCyFGGN3eBBoJIXYADwAu19upQBvgDx7us7HAEiHERmADcAD4RyS/WMQ5cwY++qjq3E9mMzRrpmwAM2aoVyg38FCwWpVCio9Xn5GYqGJDunWriJNITfVfoCkpCV59Vc0WquKll7yj2ktKYM6cqNcd12g0tUNQzsFSykXAIo+2R9y2SwCvYAQp5ZPAk57tBj2DF7MOMGOGqr1dFTExyni8fr1yP121yme68IjgdKocUEePqtoa994LjzyiPm//fmVfycxU8mzYUDntR2ws9OypFEwwnD7tu11KpUhqYyal0WiiyrkRTRINFi8O7PVktaoneYtFxSeUlakbdnx8aBlpQ8HhgAMH1PbJk/CnP6nZzZQpSlm5+O47lSdqxgxVY7xhQ7j9drjttuBjRwYOVGN4fpesLKWsNBrNOY9WGMHSsqW6ufq7+buOeeaViqZrbVERPPywUgbuisBqVdHo4USkP/ssfPml+gybTY0fFwdvvFFzS24ajaZOoXNJBcu0aYGXXUpLQ89Sa7VGPsgvP79mkiG2baviT+65B3r1gnHjYOVK5Y6r0WjOC/QMIxicThXEFhvrt1ZEeb9ApKTAiy+qpaSYGLj6apWP6cUXvfuaTOrJ3WxW/a1WpZAcDu++7sTHK2N2TZCVVRHYp9Fozju0wgiGe++F118PfLO2WtVNvrTUf5+yMlXOtZFbAYgXXlB1NRYtUgZkVwrxb75R9oiPPoKzZ2HIEJg/X5V0dcnhmVk2IQEeeqj+pibRaM4BThds4uipL7HZ87FaGpCRNojU5C5hjbl//37Gjx/P4cOHMZlMTJkyhfvuuy9CEgePVhhVsXKl35KllbDZVGnWjRu9ZyFCqCf/J56orCxcxxYsgJ074auvlEF66NCKoL97763o26MHjB0L8+apGYoQyrZw5IiavTz0kM963BqNJjqcLtjEwePzkVItT9vs+Rw8Ph8gLKVhsVj485//TI8ePSgoKKBnz54MHjyYjlXFT0UYrTCq4uabg+snhCqodNtt8O67aqbQpQscPqy8iO64QykUf1xwQVBZX7nwQpW7ysXUqSoOwjNRoEajiTpHT31ZrixcSGnj6Kkvw1IYzZo1o1kzVVw+OTmZDh06cODAAa0w6hT5+aruQzBIqXJILV+unvpHjKj6nEjgmr1oNJpax2bPD6m9OuzZs4f169dz8cWeOWBrHu0lFYjdu0MLuJMStm6tWDZyYbPVTOCeRqOpU1gtDUJqD5WzZ88yatQoXn75ZVJSUiIyZihoheGPb78NvIQUiKIiZU9Ytgy6dlX2iKQk+NWvApdA1Wg09ZqMtEGo9HgVCGElI21Q2GPbbDZGjRrFuHHj+MUvfhH2eNVBKwxf/OlPKrI5kMdTVWzfrtxmN22qSJ/xxhswcWLExNRoNHWL1OQuZDYeXj6jsFoakNl4eNheUlJKJk+eTIcOHXjggdqrOKZtGJ7s2aPyMVUV71AV8fHe3lLFxapOxqFDymVWo9Gcc6QmdwlbQXiyfPly3n//fbp06UL37t0BePrppxk6dGhEP6cqtMLwZP780HM/JSRUVg4JCco91lfEdWysyuekFYZGowmSvn374qMgadTRS1KeWK2hKYzmzVXwXUaG8lhq3lwF+V11le+04iUlyjVWo9Fo6hlaYXhy3XWhxTO8/rrKTnvkiDJo5+XBrbfCb3/rnXsqIUHZMNLTIyqyRqPRRAOtMDxp2lSlAa8Ki0Upi2HDKre5aNNGeUkNGKCWoZo0UZlkg4ka12g0mjqItmH44rbbVLGhO++sHD8RE6NSb+TmqtxOVRUN6tYN/ve/mpVVo9FoooRWGP644w6Vuvudd1RFu6FD1Usn9tNoNOcpQS1JCSGGCCG2CSF2CCEe8nE8Vggxxzi+SgiR7XZsutG+TQhxdbBj1glycuCxx+Dvf4fhw7Wy0Gg05zVVKgwhhBl4DbgG6AiMFUJ4ZryaDJySUrYBXgKeM87tCNwEdAKGAH8TQpiDHFOj0Wg0dYhgZhi9gR1Syl1SyjJgNjDSo89I4F1jey4wSAghjPbZUspSKeVuYIcxXjBjajQajaYOEYzCaA7sd9vPM9p89pFS2oF8oFGAc4MZEwAhxBQhxFohxNpjx44FIa5Go9HUIjNnQna2KnGQna32w6SkpITevXvTrVs3OnXqxKOPPhr2mNUhGKO3r6AEz5BDf338tftSVD7DGKWUM4AZALm5ubUf6qjRaDT+mDkTpkypyPywd6/aBxg3rtrDxsbGsnTpUpKSkrDZbPTt25drrrmGPtVNkFpNgplh5AEt3PazgIP++gghLEAD4GSAc4MZU6PRaOoXDz/snUOuqEi1h4EQgqSkJEBlrbXZbIhaKJgWjMJYA7QVQuQIIWJQRux5Hn3mAROM7dHAUqkSn8wDbjK8qHKAtsDqIMfUaDSa+sW+faG1h4DD4aB79+5kZGQwePDgullAybBJTAWWAFuBj6SUPwohHhdCuMrKvQk0EkLsAB4AHjLO/RH4CNgCLAbukVI6/I0Z2a+m0Wg0UaZly9DaQ8BsNrNhwwby8vJYvXo1mzdvDnvMUAkqcE9KuQhY5NH2iNt2CXCDn3OfAp4KZkyNRqOp1zz1VGUbBqgcck953QKrTWpqKgMGDGDx4sV07tw5YuMGg84lpdFoNJFi3DiVi65VK5XEtFUrtR+GwRvg2LFjnD59GoDi4mL++9//0r59+0hIHBI6NYhGo9FEknHjwlYQnhw6dIgJEybgcDhwOp3ceOONDHNPfBoltMLQaDSaOk7Xrl1Zv359bYuhl6Q0Go1GExxaYWg0Go0mKLTC0Gg0miqoC/W0g6Gm5dQKQ6PRaAIQFxfHiRMn6rzSkFJy4sQJ4qoq7BYG2uit0Wg0AcjKyiIvL4/6kPw0Li6OrKysGhtfKwyNRqMJgNVqJScnp7bFqBPoJSmNRqPRBIVWGBqNRqMJCq0wNBqNRhMUoq5b/t0RQhwD9kZwyMbA8QiOF0m0bNVDy1Y9tGzVo77I1kpKmR7ugPVKYUQaIcRaKWVubcvhCy1b9dCyVQ8tW/U432TTS1IajUajCQqtMDQajUYTFOe7wphR2wIEQMtWPbRs1UPLVj3OK9nOaxuGRqPRaILnfJ9haDQajSZItMLQaDQaTVCcMwpDCDFECLFNCLFDCPGQj+OxQog5xvFVQohst2PTjfZtQoirgx2zpmUTQgwWQqwTQmwy3q9wO+crY8wNxisjyrJlCyGK3T7/dbdzehoy7xBC/EUIIaIs2zg3uTYIIZxCiO7GsWhdt35CiO+FEHYhxGiPYxOEENuN1wS39mhdN5+yCSG6CyFWCCF+FEJsFEKMcTv2jhBit9t16x5N2YxjDrfPn+fWnmP8/bcbv4eYaMomhBjo8XsrEUJcZxyLyHULUr4HhBBbjL/dl0KIVm7HIvObk1LW+xdgBnYCrYEY4Aego0efu4HXje2bgDnGdkejfyyQY4xjDmbMKMh2EZBpbHcGDrid8xWQW4vXLRvY7Gfc1cAlgAC+AK6JpmwefboAu2rhumUDXYH3gNFu7Q2BXcZ7mrGdFuXr5k+2C4G2xnYmcAhINfbfce8b7etmHDvrZ9yPgJuM7deBX0ZbNo+/70kgIVLXLQT5Brp97i+p+F+N2G/uXJlh9AZ2SCl3SSnLgNnASI8+I4F3je25wCBDm44EZkspS6WUu4EdxnjBjFmjskkp10spDxrtPwJxQojYasgQcdn8DSiEaAakSClXSPWLfA+4rhZlGwvMqsbnhyWblHKPlHIj4PQ492rgP1LKk1LKU8B/gCHRvG7+ZJNS/iyl3G5sHwSOAmFHB0dCNn8Yf+8rUH9/UL+HqF43D0YDX0gpi6ohQ7jy/c/tc1cCrjznEfvNnSsKozmw320/z2jz2UdKaQfygUYBzg1mzJqWzZ1RwHopZalb29vGNPcP1Vy+CFe2HCHEeiHE10KIy93651UxZjRkczEGb4URjesW6rnRvG5VIoTojXqS3enW/JSx3PFSNR9cwpUtTgixVgix0rXkg/p7nzb+/tUZM1KyubgJ799buNetOvJNRs0YAp0b8m/uXFEYvv7pPf2F/fUJtT1UwpFNHRSiE/AccKfb8XFSyi7A5cbr1ijLdghoKaW8CHgA+FAIkRLkmDUtmzooxMVAkZRys9vxaF23UM+N5nULPIB68nwfuE1K6Xqang60B3qhljb+rxZkaylVqoubgZeFEBdEYMxIyea6bl2AJW7NkbhuIcknhLgFyAWer+LckL/zuaIw8oAWbvtZwEF/fYQQFqABaq3R37nBjFnTsiGEyAI+BcZLKcuf9qSUB4z3AuBD1JQ1arIZS3gnDBnWoZ5ELzT6u5f8qpXrZuD1tBfF6xbqudG8bn4xlP5C4PdSypWudinlIakoBd4m+tfNtUyGlHIXyhZ1ESq5Xqrx9w95zEjJZnAj8KmU0uYmcySuW9DyCSGuBB4GRritRkTuNxeuMaYuvFCVA3ehjNYug1Anjz73UNlA+pGx3YnKRu9dKANTlWNGQbZUo/8oH2M2NratqPXbu6IsWzpgNrZbAweAhsb+GqAPFYa0odGUzdg3of4hWtfGdXPr+w7eRu/dKONjmrEd1esWQLYY4Evgfh99mxnvAngZeDbKsqUBscZ2Y2A7htEX+BeVjd53R1M2t/aVwMBIX7cQ/h8uQj24tfVoj9hvLmTB6+oLGAr8bFywh422x1GaFiDO+GHtQHkGuN9IHjbO24abl4CvMaMpG/B7oBDY4PbKABKBdcBGlDH8FYybdxRlG2V89g/A98BwtzFzgc3GmK9iZBSI8t90ALDSY7xoXrdeKIVVCJwAfnQ7d5Ih8w7Usk+0r5tP2YBbAJvH7627cWwpsMmQ7wMgKcqyXWp8/g/G+2S3MVsbf/8dxu8hthb+ptmohyaTx5gRuW5Byvdf4Ijb325epH9zOjWIRqPRaILiXLFhaDQajaaG0QpDo9FoNEGhFYZGo9FogkIrDI1Go9EEhVYYGo1GowkKrTA0Go1GExRaYWg0Go0mKP4fE8/NCu4NPxkAAAAASUVORK5CYII=
"></div></div></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[24]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 328.6px; left: 929.6px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off">print("From cluster",center,"Selecting",len(_record),"/",len(data_pos),"samples for incremental learning")</textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true" style="bottom: 0px;"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 932.6px; margin-bottom: -16px; border-right-width: 14px; min-height: 436px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"><pre>x</pre></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"><div class="CodeMirror-selected" style="position: absolute; left: 39.2px; top: 323px; width: 890.4px; height: 17px;"></div></div><div class="CodeMirror-cursors" style="visibility: hidden;"></div><div class="CodeMirror-code" role="presentation" style=""><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">fig</span>, <span class="cm-variable">ax</span> <span class="cm-operator">=</span> <span class="cm-variable">plt</span>.<span class="cm-property">subplots</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">scatter</span> <span class="cm-operator">=</span> <span class="cm-variable">ax</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">coor_features_1</span>[:,<span class="cm-number">0</span>],<span class="cm-variable">coor_features_1</span>[:,<span class="cm-number">1</span>], <span class="cm-variable">c</span><span class="cm-operator">=</span><span class="cm-variable">kmeans</span>.<span class="cm-property">labels_</span>, <span class="cm-variable">cmap</span><span class="cm-operator">=</span><span class="cm-string">'rainbow'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">legend1</span> <span class="cm-operator">=</span> <span class="cm-variable">ax</span>.<span class="cm-property">legend</span>(<span class="cm-operator">*</span><span class="cm-variable">scatter</span>.<span class="cm-property">legend_elements</span>(), <span class="cm-variable">loc</span><span class="cm-operator">=</span><span class="cm-string">"upper left"</span>, <span class="cm-variable">title</span><span class="cm-operator">=</span><span class="cm-string">"groups"</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">ax</span>.<span class="cm-property">add_artist</span>(<span class="cm-variable">legend1</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">selected_data</span> <span class="cm-operator">=</span> []</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">for</span> <span class="cm-variable">center</span>, <span class="cm-variable">center_value</span> <span class="cm-keyword">in</span> <span class="cm-builtin">zip</span>(<span class="cm-builtin">range</span>(<span class="cm-builtin">len</span>(<span class="cm-variable">kmeans</span>.<span class="cm-property">cluster_centers_</span>)), <span class="cm-variable">kmeans</span>.<span class="cm-property">cluster_centers_</span>):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">_record</span>, <span class="cm-variable">data_pos</span> <span class="cm-operator">=</span> [], []</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-comment"># fetch features for the current center</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-keyword">for</span> <span class="cm-variable">label</span>, <span class="cm-variable">i</span> <span class="cm-keyword">in</span> <span class="cm-builtin">zip</span>(<span class="cm-variable">kmeans</span>.<span class="cm-property">labels_</span>,<span class="cm-builtin">range</span>(<span class="cm-builtin">len</span>(<span class="cm-variable">kmeans</span>.<span class="cm-property">labels_</span>))):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-keyword">if</span> <span class="cm-variable">center</span> <span class="cm-operator">==</span> <span class="cm-variable">label</span>:</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">            <span class="cm-variable">_record</span>.<span class="cm-property">append</span>(<span class="cm-variable">coor_features_1</span>[<span class="cm-variable">i</span>, :])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">            <span class="cm-variable">data_pos</span>.<span class="cm-property">append</span>(<span class="cm-variable">i</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    </span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">_record</span> <span class="cm-operator">=</span> <span class="cm-variable">finding_neighbours</span>(<span class="cm-variable">center_value</span>,<span class="cm-variable">data_pos</span>, <span class="cm-variable">_record</span>, <span class="cm-number">0.1</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">selected_data</span> <span class="cm-operator">+=</span> <span class="cm-variable">_record</span>[:,<span class="cm-number">0</span>].<span class="cm-property">tolist</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">_record</span> <span class="cm-operator">=</span> <span class="cm-variable">np</span>.<span class="cm-property">array</span>(<span class="cm-variable">_record</span>[:,<span class="cm-number">1</span>].<span class="cm-property">tolist</span>())</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-builtin">print</span>(<span class="cm-string">"From cluster"</span>,<span class="cm-variable">center</span>,<span class="cm-string">"Selecting"</span>,<span class="cm-builtin">len</span>(<span class="cm-variable">_record</span>),<span class="cm-string">"/"</span>,<span class="cm-builtin">len</span>(<span class="cm-variable">data_pos</span>),<span class="cm-string">"samples for incremental learning"</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">ax</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">_record</span>[:,<span class="cm-number">0</span>], <span class="cm-variable">_record</span>[:,<span class="cm-number">1</span>], <span class="cm-variable">cmap</span><span class="cm-operator">=</span><span class="cm-string">'rainbow'</span>, <span class="cm-variable">label</span><span class="cm-operator">=</span><span class="cm-string">"Selected data near the center "</span><span class="cm-operator">+</span><span class="cm-builtin">str</span>(<span class="cm-variable">center</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    </span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">selected_data</span> <span class="cm-operator">=</span> <span class="cm-variable">np</span>.<span class="cm-property">array</span>(<span class="cm-variable">selected_data</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">plt</span>.<span class="cm-property">legend</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 436px;"></div><div class="CodeMirror-gutters" style="display: none; height: 450px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout"><pre>From cluster 0 Selecting 22 / 229 samples for incremental learning
From cluster 1 Selecting 14 / 148 samples for incremental learning
From cluster 2 Selecting 13 / 136 samples for incremental learning
From cluster 3 Selecting 22 / 226 samples for incremental learning
</pre></div></div><div class="output_area"><div class="run_this_cell"></div><div class="prompt output_prompt"><bdi>Out[24]:</bdi></div><div class="output_subarea output_text output_result"><pre>&lt;matplotlib.legend.Legend at 0x1d7ffc0f148&gt;</pre></div></div><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_png"><img src="data:image/png;base64,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
"></div></div></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[19]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.59998px; left: 5.60001px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 714.2px; margin-bottom: -16px; border-right-width: 14px; min-height: 62px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors"><div class="CodeMirror-cursor" style="left: 5.60001px; top: 0px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-builtin">print</span>(<span class="cm-string">"=\n=\n=\n=\nclass 2\n=\n=\n=\n="</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">plt</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">coor_features_2</span>[:,<span class="cm-number">0</span>], <span class="cm-variable">coor_features_2</span>[:,<span class="cm-number">1</span>], <span class="cm-variable">label</span><span class="cm-operator">=</span><span class="cm-string">'Features of class 2'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">plt</span>.<span class="cm-property">legend</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 62px;"></div><div class="CodeMirror-gutters" style="display: none; height: 76px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout"><pre>=
=
=
=
class 2
=
=
=
=
</pre></div></div><div class="output_area"><div class="run_this_cell"></div><div class="prompt output_prompt"><bdi>Out[19]:</bdi></div><div class="output_subarea output_text output_result"><pre>&lt;matplotlib.legend.Legend at 0x1d7ffa04ec8&gt;</pre></div></div><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_png"><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXoAAAD4CAYAAADiry33AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAgAElEQVR4nO3df3Bc5X3v8ffX0mLLBizHdtpaNtgt1MTyL2Hx444TAnhimdLYIhAwl7bQSYcy1J0pbXVjUtKAyQwGptcNhRS4xTeE0rHBgKtc0jFJbcqQgGs5tmMEOJGNE0vuNDK2PDGWjX587x+7K1ars7tntSutdPR5zXi8e85znv1qpf2eZ5/nOc8xd0dERKJrXKkDEBGRoaVELyIScUr0IiIRp0QvIhJxSvQiIhFXXuoA0k2bNs1nz55d6jBEREaV3bt3H3P36UH7Rlyinz17Nk1NTaUOQ0RkVDGzX2Tap64bEZGIU6IXEYk4JXoRkYgbcX30IpJZV1cXra2tnDlzptShSIlMmDCBmTNnEovFQh+jRC8yirS2tnLeeecxe/ZszKzU4cgwc3c+/PBDWltbmTNnTujj1HUjMoqcOXOGqVOnKsmPUWbG1KlT8/5Gp0QvMsooyY9tg/n9K9GLiEScEr2I5KWsrIzFixf3/Tt8+HDedXR0dPDtb3+7+MENUkNDA9XV1TQ0NIQqP3v2bI4dO1b0OH7wgx+wZMkSFixYwJIlS9i+fXtR6tVgrIjkpaKigr179xZURzLR33333Xkd19PTQ1lZWUGvHeSpp56ivb2d8ePHF73ufEybNo3vfe97zJgxg3feeYe6ujra2toKrlctepEI27qnjaXrtzNn7assXb+drXsKTxpBenp6aGho4LLLLmPhwoU89dRTAJw6dYply5Zx6aWXsmDBAv71X/8VgLVr13Lw4EEWL15MQ0MDr7/+Or//+7/fV9+aNWv4zne+A8Rbz+vWreOzn/0sL774IgcPHmTFihUsWbKEz33uc7z//vsAvPjii8yfP59FixZx1VVXDYjR3WloaGD+/PksWLCAzZs3A7By5Uo++ugjrrjiir5tSadOneKP//iPWbBgAQsXLuSll14aUG99fT1Lliyhurqap59+uu/9uOOOO/pea8OGDQA89thjzJs3j4ULF7J69eoBddXU1DBjxgwAqqurOXPmDGfPng3/i8hALXqRiNq6p417X95PZ1cPAG0dndz78n4A6muqBl1vZ2cnixcvBmDOnDm88sorPPPMM0yePJldu3Zx9uxZli5dyvLly5k1axavvPIK559/PseOHePKK69k5cqVrF+/nnfeeafvm8Hrr7+e9TUnTJjAm2++CcCyZct48sknufjii9m5cyd3330327dvZ926dWzbto2qqio6OjoG1PHyyy+zd+9e9u3bx7Fjx7jsssu46qqraGxs5Nxzzw38lvLggw8yefJk9u+Pv28nTpwYUGbjxo186lOforOzk8suu4wbb7yRw4cP09bWxjvvvAPQF8/69ev54IMPGD9+fGCMqV566SVqamqK8i1DiV4koh7ddqAvySd1dvXw6LYDBSX6oK6b1157jZ/+9Kds2bIFgJMnT/Lzn/+cmTNn8rWvfY033niDcePG0dbWxn//93/n/Zq33HILEG9h//jHP+bLX/5y375ki3fp0qXccccd3HzzzXzpS18aUMebb77JrbfeSllZGb/xG7/B5z//eXbt2sXKlSszvu4Pf/hDNm3a1Pd8ypQpA8o89thjvPLKKwAcOXKEn//858ydO5dDhw7x53/+51x//fUsX74cgIULF3LbbbdRX19PfX19xtdtbm7mq1/9Kq+99lq2tyU0JXqRiDra0ZnX9kK4O//wD/9AXV1dv+3f+c53aG9vZ/fu3cRiMWbPnh04B7y8vJze3t6+5+llJk2aBEBvby+VlZWBre8nn3ySnTt38uqrr7J48WL27t3L1KlT+8U4mJ8r23TG119/nR/+8Ie89dZbTJw4kauvvpozZ84wZcoU9u3bx7Zt23jiiSd44YUX2LhxI6+++ipvvPEGjY2NPPjggzQ3N1Ne3j8Nt7a2csMNN/Dd736X3/md38k75iDqoxeJqBmVFXltL0RdXR3/+I//SFdXFwA/+9nP+Oijjzh58iSf/vSnicVi7Nixg1/8Ir6S7nnnncevf/3rvuMvvPBC3n33Xc6ePcvJkyf593//98DXOf/885kzZw4vvvgiEE/E+/btA+DgwYNcccUVrFu3jmnTpnHkyJF+x1511VVs3ryZnp4e2tvbeeONN7j88suz/lzLly/n8ccf73ue3nVz8uRJpkyZwsSJE3n//fd5++23ATh27Bi9vb3ceOONPPjgg/zkJz+ht7eXI0eOcM01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
"></div></div></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[20]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.59998px; left: 5.60001px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true" style="bottom: 16px;"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true" style="right: 0px; left: 0px;"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true" style="height: 16px; width: 16px;"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 999.8px; margin-bottom: -16px; border-right-width: 14px; min-height: 402px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors"><div class="CodeMirror-cursor" style="left: 5.60001px; top: 0px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation" style=""><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">cluster</span> <span class="cm-keyword">import</span> <span class="cm-variable">KMeans</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">kmeans_2</span> <span class="cm-operator">=</span> <span class="cm-variable">KMeans</span>(<span class="cm-variable">n_clusters</span><span class="cm-operator">=</span><span class="cm-number">2</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">kmeans_2</span>.<span class="cm-property">fit</span>(<span class="cm-variable">coor_features_2</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">fig</span>, <span class="cm-variable">ax</span> <span class="cm-operator">=</span> <span class="cm-variable">plt</span>.<span class="cm-property">subplots</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">scatter</span> <span class="cm-operator">=</span> <span class="cm-variable">ax</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">coor_features_2</span>[:,<span class="cm-number">0</span>],<span class="cm-variable">coor_features_2</span>[:,<span class="cm-number">1</span>], <span class="cm-variable">c</span><span class="cm-operator">=</span><span class="cm-variable">kmeans_2</span>.<span class="cm-property">labels_</span>, <span class="cm-variable">cmap</span><span class="cm-operator">=</span><span class="cm-string">'rainbow'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-comment"># produce a legend with the unique colors from the scatter</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">legend1</span> <span class="cm-operator">=</span> <span class="cm-variable">ax</span>.<span class="cm-property">legend</span>(<span class="cm-operator">*</span><span class="cm-variable">scatter</span>.<span class="cm-property">legend_elements</span>(), <span class="cm-variable">loc</span><span class="cm-operator">=</span><span class="cm-string">"upper left"</span>, <span class="cm-variable">title</span><span class="cm-operator">=</span><span class="cm-string">"groups"</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">ax</span>.<span class="cm-property">add_artist</span>(<span class="cm-variable">legend1</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">num_centers</span> <span class="cm-operator">=</span> <span class="cm-builtin">len</span>(<span class="cm-variable">kmeans_2</span>.<span class="cm-property">cluster_centers_</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">centers_center</span> <span class="cm-operator">=</span> [<span class="cm-builtin">sum</span>(<span class="cm-variable">kmeans_2</span>.<span class="cm-property">cluster_centers_</span>[:,<span class="cm-number">0</span>])<span class="cm-operator">/</span><span class="cm-variable">num_centers</span>, <span class="cm-builtin">sum</span>(<span class="cm-variable">kmeans_2</span>.<span class="cm-property">cluster_centers_</span>[:,<span class="cm-number">1</span>])<span class="cm-operator">/</span><span class="cm-variable">num_centers</span>]</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-comment"># plot clusters</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">ax</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">coor_features_2</span>[:,<span class="cm-number">0</span>],<span class="cm-variable">coor_features_2</span>[:,<span class="cm-number">1</span>], <span class="cm-variable">c</span><span class="cm-operator">=</span><span class="cm-variable">kmeans_2</span>.<span class="cm-property">labels_</span>, <span class="cm-variable">cmap</span><span class="cm-operator">=</span><span class="cm-string">'rainbow'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-comment"># plot centers center</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">ax</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">centers_center</span>[<span class="cm-number">0</span>], <span class="cm-variable">centers_center</span>[<span class="cm-number">1</span>], <span class="cm-variable">cmap</span><span class="cm-operator">=</span><span class="cm-string">'rainbow'</span>, <span class="cm-variable">label</span><span class="cm-operator">=</span><span class="cm-string">"Centers center"</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-comment"># plot centers</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">ax</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">kmeans_2</span>.<span class="cm-property">cluster_centers_</span>[:,<span class="cm-number">0</span>], <span class="cm-variable">kmeans_2</span>.<span class="cm-property">cluster_centers_</span>[:,<span class="cm-number">1</span>],<span class="cm-variable">cmap</span><span class="cm-operator">=</span><span class="cm-string">'rainbow'</span>, <span class="cm-variable">marker</span><span class="cm-operator">=</span><span class="cm-string">"+"</span>, <span class="cm-variable">label</span><span class="cm-operator">=</span><span class="cm-string">"centers"</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">for</span> <span class="cm-variable">n</span> <span class="cm-keyword">in</span> <span class="cm-builtin">range</span>(<span class="cm-builtin">len</span>(<span class="cm-variable">kmeans_2</span>.<span class="cm-property">cluster_centers_</span>)):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-builtin">print</span>(<span class="cm-string">"cluster"</span>,<span class="cm-variable">n</span> ,<span class="cm-string">"contains"</span>, <span class="cm-builtin">sum</span>(<span class="cm-variable">kmeans_2</span>.<span class="cm-property">labels_</span><span class="cm-operator">==</span><span class="cm-variable">n</span>), <span class="cm-string">"features"</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-builtin">print</span>(<span class="cm-string">"center of centers : X="</span>,<span class="cm-variable">centers_center</span>[<span class="cm-number">0</span>],<span class="cm-string">"Y="</span>, <span class="cm-variable">centers_center</span>[<span class="cm-number">1</span>])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">plt</span>.<span class="cm-property">legend</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 402px;"></div><div class="CodeMirror-gutters" style="display: none; height: 416px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout"><pre>cluster 0 contains 71 features
cluster 1 contains 54 features
center of centers : X= 0.016173890112167105 Y= 0.02063570440697344
</pre></div></div><div class="output_area"><div class="run_this_cell"></div><div class="prompt output_prompt"><bdi>Out[20]:</bdi></div><div class="output_subarea output_text output_result"><pre>&lt;matplotlib.legend.Legend at 0x1d7ffa88f48&gt;</pre></div></div><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_png"><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXoAAAD4CAYAAADiry33AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAgAElEQVR4nO3deXxU1dnA8d+TBMIiiyxuREgQBNmXgOLCqoCgRivWoLUsVorVqn0rvqi1bq0bKi5oAQuKioKgAq74VsVqi0hQkE0kIEhAJQKyhyTkef84EzKZzEwmk0kmmTzfz2c+mXvPuec+mUyeuXPuueeKqmKMMSZ2xUU7AGOMMRXLEr0xxsQ4S/TGGBPjLNEbY0yMs0RvjDExLiHaAfhq1qyZJicnRzsMY4ypVlasWPGzqjb3V1blEn1ycjIZGRnRDsMYY6oVEdkaqMy6bowxJsZZojfGmBhnid4YY2Jcleuj9ycvL4+srCxycnKiHUqp6tSpQ1JSErVq1Yp2KMaUqjr9bxknnBxTLRJ9VlYWDRo0IDk5GRGJdjgBqSq7du0iKyuLlJSUaIdjTKmqy/+WccLNMdWi6yYnJ4emTZtW+TeiiNC0aVM7OjLVRnX53zJOuDmmWiR6oNq8EatLnMYUsvds9RLO36vaJHpjjDHhsURvjImqH3/8kfT0dE477TQ6dOjAsGHD+Pbbb8Nq64UXXmDHjh0RjjBytmzZwiuvvFLp+43pRJ+fnx/tEIwxQagql112Gf3792fTpk2sW7eOBx54gJ9++ims9sJJ9JWZJ8JJ9EePHi33fqt1or///vtp3749F1xwASNHjuTRRx+lf//+3HHHHfTr148nn3ySrVu3MmjQILp06cKgQYP4/vvvARg9ejTz588/1tZxxx0HwJIlS+jbty+XXXYZHTp0YPz48RQUFHD06FFGjx5Np06d6Ny5M5MnT47K72xMNC34ajvnPPQRKRPf4ZyHPmLBV9vL1d7HH39MrVq1GD9+/LF13bp147zzzgNg0qRJ9OrViy5dunD33XcDLlmeccYZXHfddXTs2JHBgwdz+PBh5s+fT0ZGBldffTXdunXj8OHDrFixgn79+tGzZ0+GDBnCDz/8AFAiT8ybN49OnTrRtWtX+vbt6zfWRx55hM6dO9O1a1cmTpwIwKZNmxg6dCg9e/bkvPPO45tvvgFcfrnppps4++yzad269bFcM3HiRD799FO6devG5MmTOXr0KBMmTDj2O06bNg1weWjAgAFcddVVdO7cuVyvMeA+UavSo2fPnupr3bp1JdYtX75cu3btqocOHdJ9+/ZpmzZtdNKkSdqvXz+9/vrrj9W76KKL9IUXXlBV1RkzZmhaWpqqqo4aNUrnzZt3rF79+vVVVfXjjz/WxMRE3bRpk+bn5+v555+v8+bN04yMDD3//POP1d+zZ0+JmILFa0xVVJb36ptfZmn7v7ynrf737WOP9n95T9/8Mivs/T/55JN6yy23+C1bvHixXnfddVpQUKBHjx7V4cOH6yeffKLfffedxsfH61dffaWqqldccYW+9NJLqqrar18/Xb58uaqq5ubmap8+fXTnzp2qqjpnzhwdM2bMsXreeaJTp06aleV+D3//2++++6726dNHDx48qKqqu3btUlXVgQMH6rfffquqqp9//rkOGDBAVV1+GTFihB49elTXrl2rp512mqq6/DJ8+PBj7U6bNk3vv/9+VVXNycnRnj176ubNm/Xjjz/WevXq6ebNm/2+Nv7+bkCGBsir1WIcvT+fffYZaWlp1K1bF4CLL774WNmVV1557PnSpUt54403ALjmmmu47bbbSm27d+/etG7dGoCRI0fy2WefMWjQIDZv3swf//hHhg8fzuDBgyP56xhT5U1avIHDecW7EQ7nHWXS4g1c2r1FxPf3wQcf8MEHH9C9e3cADhw4wMaNG2nZsiUpKSl069YNgJ49e7Jly5YS22/YsIE1a9ZwwQUXAK4L5OSTTz5W7p0nzjnnHEaPHs2vf/1rfvWrX5Vo61//+hdjxoyhXr16ADRp0oQDBw7w3//+lyuuuOJYvSNHjhx7fumllxIXF0eHDh0CdkV98MEHfP3118eO+Pfu3cvGjRupXbs2vXv3jtj1ONU20WuQm5rXr18/YFnh0KSEhAQKCgqOtZWbm1uijvfy8ccfz6pVq1i8eDHPPPMMr732GjNnzizPr2BMtbLjl8NlWh+Kjh07FutC9aaq3H777fz+978vtn7Lli0kJiYeW46Pj+fw4ZIxqCodO3Zk6dKlftv3zhNTp05l2bJlvPPOO3Tr1o2VK1fStGnTYm355oWCggIaN27MypUr/bbvHWOgfKWqPP300wwZMqTY+iVLlgTNY2VVbfvozz33XN56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
"></div></div></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div><div class="cell code_cell rendered selected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[25]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython CodeMirror-focused"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 345.6px; left: 39.2px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true" style="bottom: 0px;"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 974.6px; margin-bottom: -16px; border-right-width: 14px; min-height: 419px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors" style=""><div class="CodeMirror-cursor" style="left: 39.2px; top: 340px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation" style=""><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">fig</span>, <span class="cm-variable">ax</span> <span class="cm-operator">=</span> <span class="cm-variable">plt</span>.<span class="cm-property">subplots</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">scatter</span> <span class="cm-operator">=</span> <span class="cm-variable">ax</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">coor_features_2</span>[:,<span class="cm-number">0</span>],<span class="cm-variable">coor_features_2</span>[:,<span class="cm-number">1</span>], <span class="cm-variable">c</span><span class="cm-operator">=</span><span class="cm-variable">kmeans_2</span>.<span class="cm-property">labels_</span>, <span class="cm-variable">cmap</span><span class="cm-operator">=</span><span class="cm-string">'rainbow'</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">legend1</span> <span class="cm-operator">=</span> <span class="cm-variable">ax</span>.<span class="cm-property">legend</span>(<span class="cm-operator">*</span><span class="cm-variable">scatter</span>.<span class="cm-property">legend_elements</span>(), <span class="cm-variable">loc</span><span class="cm-operator">=</span><span class="cm-string">"lower right"</span>, <span class="cm-variable">title</span><span class="cm-operator">=</span><span class="cm-string">"groups"</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">ax</span>.<span class="cm-property">add_artist</span>(<span class="cm-variable">legend1</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">selected_data</span> <span class="cm-operator">=</span> []</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-keyword">for</span> <span class="cm-variable">center</span>, <span class="cm-variable">center_value</span> <span class="cm-keyword">in</span> <span class="cm-builtin">zip</span>(<span class="cm-builtin">range</span>(<span class="cm-builtin">len</span>(<span class="cm-variable">kmeans_2</span>.<span class="cm-property">cluster_centers_</span>)), <span class="cm-variable">kmeans_2</span>.<span class="cm-property">cluster_centers_</span>):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">_record</span>, <span class="cm-variable">data_pos</span> <span class="cm-operator">=</span> [], []</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-comment"># fetch features for the current center</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-keyword">for</span> <span class="cm-variable">label</span>, <span class="cm-variable">i</span> <span class="cm-keyword">in</span> <span class="cm-builtin">zip</span>(<span class="cm-variable">kmeans_2</span>.<span class="cm-property">labels_</span>, <span class="cm-builtin">range</span>(<span class="cm-builtin">len</span>(<span class="cm-variable">kmeans_2</span>.<span class="cm-property">labels_</span>))):</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">        <span class="cm-keyword">if</span> <span class="cm-variable">center</span> <span class="cm-operator">==</span> <span class="cm-variable">label</span>:</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">            <span class="cm-variable">_record</span>.<span class="cm-property">append</span>(<span class="cm-variable">coor_features_2</span>[<span class="cm-variable">i</span>, :])</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">            <span class="cm-variable">data_pos</span>.<span class="cm-property">append</span>(<span class="cm-variable">i</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    </span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">_record</span> <span class="cm-operator">=</span> <span class="cm-variable">finding_neighbours</span>(<span class="cm-variable">center_value</span>, <span class="cm-variable">data_pos</span>, <span class="cm-variable">_record</span>, <span class="cm-number">0.1</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">selected_data</span> <span class="cm-operator">+=</span> <span class="cm-variable">_record</span>[:,<span class="cm-number">0</span>].<span class="cm-property">tolist</span>()</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">_record</span> <span class="cm-operator">=</span> <span class="cm-variable">np</span>.<span class="cm-property">array</span>(<span class="cm-variable">_record</span>[:,<span class="cm-number">1</span>].<span class="cm-property">tolist</span>())</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-builtin">print</span>(<span class="cm-string">"From cluster"</span>, <span class="cm-variable">center</span>, <span class="cm-string">"Selecting"</span>, <span class="cm-builtin">len</span>(<span class="cm-variable">_record</span>),<span class="cm-string">"/"</span>, <span class="cm-builtin">len</span>(<span class="cm-variable">data_pos</span>), <span class="cm-string">"samples for incremental learning"</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    <span class="cm-variable">ax</span>.<span class="cm-property">scatter</span>(<span class="cm-variable">_record</span>[:,<span class="cm-number">0</span>], <span class="cm-variable">_record</span>[:,<span class="cm-number">1</span>], <span class="cm-variable">cmap</span><span class="cm-operator">=</span><span class="cm-string">'rainbow'</span>, <span class="cm-variable">label</span><span class="cm-operator">=</span><span class="cm-string">"Selected data near the center "</span><span class="cm-operator">+</span><span class="cm-builtin">str</span>(<span class="cm-variable">center</span>))</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation">    </span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">selected_data</span> <span class="cm-operator">=</span> <span class="cm-variable">np</span>.<span class="cm-property">array</span>(<span class="cm-variable">selected_data</span>)</span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span class="cm-variable">plt</span>.<span class="cm-property">legend</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 419px;"></div><div class="CodeMirror-gutters" style="display: none; height: 433px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout"><pre>From cluster 0 Selecting 7 / 71 samples for incremental learning
From cluster 1 Selecting 5 / 54 samples for incremental learning
</pre></div></div><div class="output_area"><div class="run_this_cell"></div><div class="prompt output_prompt"><bdi>Out[25]:</bdi></div><div class="output_subarea output_text output_result"><pre>&lt;matplotlib.legend.Legend at 0x1d7ffd7f248&gt;</pre></div></div><div class="output_area"><div class="run_this_cell"></div><div class="prompt"></div><div class="output_subarea output_png"><img src="data:image/png;base64,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
"></div></div></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div><div class="cell code_cell rendered unselected" tabindex="2"><div class="input"><div class="prompt_container"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[&nbsp;]:</div><div class="run_this_cell" title="Run this cell"><i class="fa-step-forward fa"></i></div></div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area" aria-label="Edit code here"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 5.59998px; left: 5.60001px;"><textarea style="position: absolute; bottom: -1em; padding: 0px; width: 1px; height: 1em; outline: currentcolor none medium;" autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" wrap="off"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1" draggable="true"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 8.60001px; margin-bottom: -16px; border-right-width: 14px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div style="position: relative; outline: currentcolor none medium;" role="presentation"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors"><div class="CodeMirror-cursor" style="left: 5.60001px; top: 0px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation"><span cm-text="">​</span></span></pre></div></div></div></div></div><div style="position: absolute; height: 14px; width: 1px; border-bottom: 0px solid transparent; top: 28px;"></div><div class="CodeMirror-gutters" style="display: none; height: 42px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"></div><div style="display: none;" class="btn btn-default output_collapsed" title="click to expand output">. . .</div></div></div></div><div class="end_space"></div></div>
        <div id="tooltip" class="ipython_tooltip" style="display:none"><div class="tooltipbuttons"><a href="#" role="button" class="ui-button"><span class="ui-icon ui-icon-close">Close</span></a><a href="#" role="button" class="ui-button" id="expanbutton" title="Grow the tooltip vertically (press shift-tab twice)"><span class="ui-icon ui-icon-plus">Expand</span></a><a href="#" role="button" class="ui-button" title="show the current docstring in pager (press shift-tab 4 times)"><span class="ui-icon ui-icon-arrowstop-l-n">Open in Pager</span></a><a href="#" role="button" class="ui-button" title="Tooltip will linger for 10 seconds while you type" style="display: none;"><span class="ui-icon ui-icon-clock">Close</span></a></div><div class="pretooltiparrow"></div><div class="tooltiptext smalltooltip"></div></div>
    </div>
</div>



</div>



<div id="pager" class="ui-resizable">
    <div id="pager-contents">
        <div id="pager-container" class="container"></div>
    </div>
    <div id="pager-button-area"><a role="button" title="Open the pager in an external window" class="ui-button"><span class="ui-icon ui-icon-extlink"></span></a><a role="button" title="Close the pager" class="ui-button"><span class="ui-icon ui-icon-close"></span></a></div>
<div class="ui-resizable-handle ui-resizable-n" style="z-index: 90;"></div></div>






<script type="text/javascript">
    sys_info = {"notebook_version": "6.0.1", "notebook_path": "E:\\etud\\lib\\site-packages\\notebook", "commit_source": "", "commit_hash": "", "sys_version": "3.7.4 (default, Aug  9 2019, 18:34:13) [MSC v.1915 64 bit (AMD64)]", "sys_executable": "E:\\etud\\python.exe", "sys_platform": "win32", "platform": "Windows-10-10.0.18362-SP0", "os_name": "nt", "default_encoding": "utf-8"};
</script>

<script src="test%20-%20Jupyter%20Notebook_files/encoding.js" charset="utf-8"></script>

<script src="test%20-%20Jupyter%20Notebook_files/main.js" type="text/javascript" charset="utf-8"></script>



<script type="text/javascript">
  function _remove_token_from_url() {
    if (window.location.search.length <= 1) {
      return;
    }
    var search_parameters = window.location.search.slice(1).split('&');
    for (var i = 0; i < search_parameters.length; i++) {
      if (search_parameters[i].split('=')[0] === 'token') {
        // remote token from search parameters
        search_parameters.splice(i, 1);
        var new_search = '';
        if (search_parameters.length) {
          new_search = '?' + search_parameters.join('&');
        }
        var new_url = window.location.origin + 
                      window.location.pathname + 
                      new_search + 
                      window.location.hash;
        window.history.replaceState({}, "", new_url);
        return;
      }
    }
  }
  _remove_token_from_url();
</script>


<div style="position: absolute; width: 0px; height: 0px; overflow: hidden; padding: 0px; border: 0px none; margin: 0px;"><div id="MathJax_Font_Test" style="position: absolute; visibility: hidden; top: 0px; left: 0px; width: auto; min-width: 0px; max-width: none; padding: 0px; border: 0px none; margin: 0px; white-space: nowrap; text-align: left; text-indent: 0px; text-transform: none; line-height: normal; letter-spacing: normal; word-spacing: normal; font-size: 40px; font-weight: normal; font-style: normal; font-size-adjust: none;"></div></div></body></html>
