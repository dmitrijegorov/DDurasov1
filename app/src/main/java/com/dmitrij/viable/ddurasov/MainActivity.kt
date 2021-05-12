package com.dmitrij.viable.ddurasov

import android.Manifest
import android.app.Activity
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.dmitrij.viable.ddurasov.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private val REQUEST_CODE = 210;


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)


        // permissions
        ActivityCompat.requestPermissions(
            this, arrayOf(
                Manifest.permission.CAMERA, Manifest.permission.INTERNET
            ), REQUEST_CODE
        )

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        overridePendingTransition(R.anim.change_fade_fragment, R.anim.change_fade_out_fragment)


    }
}