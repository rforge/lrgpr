<?php
	if( isset($_SESSION['views']) ){
	    $_SESSION['views']++;
	}else{
		$_SESSION['views'] = 0;
	}
	echo $_SESSION['views'];
?>	
