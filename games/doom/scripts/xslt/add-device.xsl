<?xml version="1.0" encoding="UTF-8"?>
<!--
  add-device.xsl - Add __device__ qualifier to function definitions

  Doom GPU transpilation - bare bones scaffold
-->
<xsl:stylesheet version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:src="http://www.srcML.org/srcML/src"
    xmlns:cpp="http://www.srcML.org/srcML/cpp">

  <xsl:output method="xml" indent="no"/>

  <!-- Identity transform -->
  <xsl:template match="@*|node()">
    <xsl:copy>
      <xsl:apply-templates select="@*|node()"/>
    </xsl:copy>
  </xsl:template>

  <!-- Add __device__ to function definitions (have block) -->
  <xsl:template match="src:function[src:block]">
    <xsl:copy>
      <xsl:apply-templates select="@*"/>
      <src:specifier>__device__</src:specifier>
      <xsl:text> </xsl:text>
      <xsl:apply-templates select="node()"/>
    </xsl:copy>
  </xsl:template>

  <!-- Exclude main() -->
  <xsl:template match="src:function[src:name='main'][src:block]">
    <xsl:copy>
      <xsl:apply-templates select="@*|node()"/>
    </xsl:copy>
  </xsl:template>

</xsl:stylesheet>
