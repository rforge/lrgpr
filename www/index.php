
<!-- This is the project specific website template -->
<!-- It can be changed as liked or replaced by other content -->


<div id="container" style="width:500px">

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
	<link href="http://<?php echo $themeroot; ?>styles/estilo1.css" rel="stylesheet" type="text/css" />
  </head>

<body>

<!-- R-Forge Logo -->
<table border="0" width="100%" cellspacing="0" cellpadding="0">
<tr><td>
<a href="http://r-forge.r-project.org/"><img src="http://<?php echo $themeroot; ?>/imagesrf/logo.png" border="0" alt="R-Forge Logo" /> </a> </td> </tr>
</table>


<!-- get project title  -->
<!-- own website starts here, the following may be changed as you like -->

<?php if ($handle=fopen('http://'.$domain.'/export/projtitl.php?group_name='.$group_name,'r')){
$contents = '';
while (!feof($handle)) {
	$contents .= fread($handle, 8192);
}
fclose($handle);
echo $contents; } ?>

<!-- end of project description -->
<div id="content" style="background-color:#EEEEEE;height:400px;width:800px;float:left;">

	<p> <b> lrgpr: Low Rank Gaussian Process Regression </b> </p>

	<b>lrgpr</b> is a high-peformance, user-friendly R interface for evaluating linear mixed models. 

	<p> <a href="./docs/lrgpr.pdf">Documentation</a> gives a thorough description of the advantages and applications of <b>lrgpr</b>.</p>

	<p> See the <a href="./docs/lrgpr-manual.pdf">manual</a> for details about available functions.</p>

	<p> Download package: <a href="./docs/lrgpr_0.0.1.tar.gz">lrgpr_0.0.1.tar.gz</a> </p>

	<p> Citation:</p>

	<p>
	 Hoffman, G.E., Mezey, J.G., Schadt, E.E. 2014. lrgpr: Interactive linear mixed model analysis of genome-wide association studies with composite hypothesis testing and regression diagnostics. In preparation. </p>
	 </p>
	 
	<p> <a href="http://<?php echo $domain; ?>/projects/<?php echo $group_name; ?>/">Project summary page</a> </p>

</div>

<div id="footer" style="background-color:#FFA500;clear:both;text-align:center;"> 
 </div>

</body>
</html>
