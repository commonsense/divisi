from numpy import *

svg_header = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->
<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   width="744.09448819"
   height="1052.3622047"
   id="svg2"
   sodipodi:version="0.32"
   inkscape:version="0.46"
   sodipodi:docname="test.svg"
   inkscape:output_extension="org.inkscape.output.svg.inkscape">
  <defs
     id="defs4">
    <inkscape:perspective
       sodipodi:type="inkscape:persp3d"
       inkscape:vp_x="0 : 526.18109 : 1"
       inkscape:vp_y="0 : 1000 : 0"
       inkscape:vp_z="744.09448 : 526.18109 : 1"
       inkscape:persp3d-origin="372.04724 : 350.78739 : 1"
       id="perspective10" />
  </defs>
  <sodipodi:namedview
     id="base"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     gridtolerance="10000"
     guidetolerance="10"
     objecttolerance="10"
     inkscape:pageopacity="0.0"
     inkscape:pageshadow="2"
     inkscape:zoom="3.2240805"
     inkscape:cx="98.734628"
     inkscape:cy="916.53216"
     inkscape:document-units="px"
     inkscape:current-layer="layer1"
     showgrid="false"
     inkscape:window-width="1150"
     inkscape:window-height="719"
     inkscape:window-x="62"
     inkscape:window-y="232" />
  <metadata
     id="metadata7">
    <rdf:RDF>
      <cc:Work
         rdf:about="">
        <dc:format>image/svg+xml</dc:format>
        <dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
      </cc:Work>
    </rdf:RDF>
  </metadata>
  <g
     inkscape:label="Layer 1"
     inkscape:groupmode="layer"
     id="layer1">
"""

svg_footer = """
</g>
</svg>
"""

def svg_text(x, y, text):
    return """
    <text
       xml:space="preserve"
       style="font-size:9px;font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;fill:#000000;fill-opacity:1;stroke:none;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1;font-family:Sans;-inkscape-font-specification:Sans"
       x="%(x)s"
       y="%(y)s"
       id="text12"><tspan
         sodipodi:role="line"
         id="tspan14"
         x="%(x)s"
         y="%(y)s">%(text)s</tspan></text>
    """ % locals()

def svg_diamond(x, y, color='#ff0000'):
    xt = x * 0.7071 + y * 0.7071
    yt = - x * 0.7071 + y * 0.7071
    return """
    <rect
       style="fill:%(color)s;fill-opacity:1;stroke:none;stroke-width:1;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       id="rect2399"
       width="6.0455203"
       height="6.0455203"
       x="%(xt)s"
       y="%(yt)s"
       transform="matrix(0.7071068,0.7071068,-0.7071068,0.7071068,0,0)" />
    """ % locals()

def color_scale(val):
    hval = int(128.0 + val*4000)
    if hval < 0: return 0
    if hval > 255: return 255
    else: return hval

def xmlspecialchars(text):
    return text.replace(u"&", u"&amp;")\
           .replace(u"<", u"&lt;")\
           .replace(u">", u"&gt;")\
           .replace(u"'", u"&apos;")\
           .replace(u'"', u"&quot;")

def make_color(red, green, blue):
    return "#%x%x%x" % (color_scale(red), color_scale(green), color_scale(blue))

def output_svg(u, filename, concepts=None, xscale=1000, yscale=1000, min=0):
    if concepts is None: concepts = u.label_list(0)
    out = open(filename, 'w')
    try:
        print >> out, svg_header
        diamonds = []
        for concept in concepts:
            x, y = u[concept, 0], u[concept, 1]
            diamonds.append(svg_diamond(x*xscale+350, y*yscale+450,
            make_color(u[concept, 2], u[concept, 3], u[concept, 4])))
            if abs(x) > min or abs(y) > min:
                print >> out, svg_text(x*xscale+350, y*yscale+450, xmlspecialchars(concept)).encode('utf-8')
        print >> out, '\n'.join(diamonds).encode('utf-8')
        print >> out, svg_footer
    finally:
        out.close()
