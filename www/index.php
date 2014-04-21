
<!-- This is the project specific website template -->
<!-- It can be changed as liked or replaced by other content -->

<?php

$domain=ereg_replace('[^\.]*\.(.*)$','\1',$_SERVER['HTTP_HOST']);
$group_name=ereg_replace('([^\.]*)\..*$','\1',$_SERVER['HTTP_HOST']);
$themeroot='r-forge.r-project.org/themes/rforge/';

echo '<?xml version="1.0" encoding="UTF-8"?>';
?>

<!DOCTYPE html
	PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
	"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en   ">

  <head>
	<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
	<title><?php echo $group_name; ?></title>
	<!--<link href="http://<?php echo $themeroot; ?>styles/estilo1.css" rel="stylesheet" type="text/css" /> -->
	<link href="vegan.css" rel="stylesheet" type="text/css" />
  </head>

<body>

<!-- R-Forge Logo -->
<table border="0" width="100%" cellspacing="0" cellpadding="0">
<tr><td>
<a href="http://r-forge.r-project.org/"><img src="http://<?php echo $themeroot; ?>/imagesrf/logo.png" border="0" alt="R-Forge Logo" height=30 /> </a> 
<a href="http://icahn.mssm.edu/departments-and-institutes/genomics"> <img src="MSSMLogo.png" height=30 align="right" /> </a> 
</td> </tr>
</table>


<!-- get project title  -->
<!-- own website starts here, the following may be changed as you like -->

<!--<?php if ($handle=fopen('http://'.$domain.'/export/projtitl.php?group_name='.$group_name,'r')){
$contents = '';
while (!feof($handle)) {
	$contents .= fread($handle, 8192);
}
fclose($handle);
echo $contents; } ?> -->

<!-- end of project description -->


<!-- Title -->

<font size=4>
<p> <b> lrgpr: Low Rank Gaussian Process Regression </b> </p>
</font>

<p>
<b>lrgpr</b> is a high-peformance, user-friendly R interface for evaluating linear mixed models.  This package is designed for interactive, exploratory analysis of large genome-wide assocation studies (GWAS) using linear mixed models to account for the confounding effects of kinship and population structure. 
</p> 
<p>
The package also provides simple interfaces for standard linear and logistic regression models. It allows fitting millions of regression models on a desktop computer by using an efficient implementation, parallelization and out-of-core computing for datasets that are too large to fit in main memory.  
</p>

<ul>
<li> <a href="./docs/lrgpr.pdf"><strong>Tutorial</strong></a>: thorough demonstration of the functionality and applications of <b>lrgpr</b>	</li>

<li> <a href="./docs/lrgpr-manual.pdf"><strong>Manual</strong></a>: details about available functions</li>

<li> <a href="./docs/lrgpr_0.0.6.tar.gz"><strong>Download package</strong></a>: v0.0.6 </li>

<li> <a href="http://<?php echo $domain; ?>/projects/<?php echo $group_name; ?>/"><strong>Project summary page</strong></a> </li>

<br>
<li> <strong>Citation</strong>:</li>

 Hoffman, G.E., Mezey, J.G., Schadt, E.E. (2014) lrgpr: Interactive linear mixed model analysis of genome-wide association studies with composite hypothesis testing and regression diagnostics in R. In preparation. </p>

<li> <strong>Related work</strong>:</li>

 Hoffman, G.E. (2013) Correcting for Population Structure and Kinship Using the Linear Mixed Model: Theory and Extensions. PLoS ONE 8(10): e75707. <a href="http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0075707">doi:10.1371/journal.pone.0075707</a> </p>

</ul>

<br>
<br>
<font size=1>
<p> Contact: gabriel [dot] hoffman [at] mssm.edu </p>
</font>

<hr />
<font size="1">
<div id="textbox">
  <p class="alignleft">Gabriel E. Hoffman.  Updated April 18, 2014</p>
  <!--<p class="alignright">
  	<?
	echo "UPV: ";
	include_once "counter.php"
	?></p>-->
</div>
</font>

</body>
</html>
