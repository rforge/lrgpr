
<?php
// Inits
$file = "counts.txt";
$cookie_namee='mycounterr-456';

// File, created if !exists
$fh = fopen($file, 'w+');
$fh2 = fopen("addr.txt", 'w+');

// Get the count, 0 if the file is empty or just created
$count = (int)fgets($fh);

//if cookie isn't already set,then increase counts 
//by one + save ip, and set a cookie to pc...
if (!isset($_COOKIE[$cookie_namee])) {
    // Increment and write the new count
    fwrite($fh, ++$count);
    setcookie($cookie_namee,"Checked",time()+111400);

    fwrite($fh2, $_SERVER['REMOTE_HOST']);
}

fclose($fh);
fclose($fh2);
echo $count;
?>