#version 330 core

#define PI 3.14159265359

// fragment position and normal of the fragment, in WORLD coordinates
in vec3 w_position, w_normal;   // in world coodinates
in vec3 position;

// world camera position
uniform vec3 w_camera_position;

// texturing
in vec2 frag_tex_coords;
// skybox sides textures
uniform sampler2D diffuse_map;
// terrain textures
uniform sampler2D grass;
uniform sampler2D rock;
uniform sampler2D sand;
// lava texture
uniform sampler2D lava;
// wood texture
uniform sampler2D wood;
// normal maps
uniform sampler2D normalmap;


// surface identifier
uniform int surface_type;
// surfaces types constants
uniform int terrain_type;
uniform int water_type;
uniform int lava_type;
uniform int bubble_type;
uniform int cloud_type;
uniform int projection_type;
uniform int smoke_type;
uniform int skybox_type; 
uniform int wood_type;
uniform int cactus_type;

// material properties
uniform vec3 k_d;
uniform vec3 k_a;
uniform vec3 k_s;
uniform vec3 k_l;
uniform float s;

// offsets for surfaces
uniform float terrain_offset;
uniform float lava_offset;

// directional light
uniform vec3 light_dir;
uniform vec3 light_color;

// time
uniform float time;

// output fragment color for OpenGL
out vec4 out_color;

// noise at vertex (not pixel !)
//in float noise;

// whether to light or not (ray marched)
//in float shadow; // used as a bool (<0.5 <=> false)

// foam effect on water
in float foam;

// depth to lava for water effect, etc
in float lava_depth;

// fogging coefficient, keyboard controlled (left/right)
uniform float fog_coef;

// ray marching constants
#define DIST_MIN 1e-2
#define DIST_MAX 15
#define RAY_MARCH_STEPS 50
#define RAY_MARCH_PRECI 1.2e-2

// used in generating perlin noise
const int N_perlin = 10;
#define PERLIN_SCALE 3
#define PERLIN_ITER 4

// square euclidean distance
float distance2(vec3 v1, vec3 v2)
{
    vec3 delta = v1 - v2;
    return dot(delta, delta);
}

// Apply quaternion q to vector v
vec3 rotate(vec3 v, vec4 q)
{ 
	return v + 2.0 * cross(cross(v, q.xyz) + v*q.w, q.xyz);
} 

// Pseudo random generator
float rand(vec2 co)
{
    return fract(sin(dot(co, vec2(9.745, 65.24))) * 458.5453);
}

// 2D perlin noise, iterated
float noise2(const in vec2 p)
{
    float scale = PERLIN_SCALE;
    float amp = 0.4;

    float result = 0.;

    for(int i = 0; i < PERLIN_ITER; i++)
    {
        vec2 xy = mod(p, scale)/scale;
        vec2 uv = xy * vec2(N_perlin);
        vec2 fl = floor(uv);
        vec2 ce = fl + vec2(1);
        vec2 interp = mod(uv, 1);

        if(ce.x >= N_perlin)
            ce.x = 0;
        if(ce.y >= N_perlin)
            ce.y = 0;

        result += amp * mix(mix(rand(fl), rand(vec2(ce.x, fl.y)), interp.x),
                        mix(rand(vec2(fl.x, ce.y)), rand(ce), interp.x),
                        interp.y);

        scale /= 2.3;
        amp /= 1.7;
    }

    return result;
}

// 2D gaussian (volcano base shape)
float gaussian(const in vec3 p, const float max_y, const float div)
{
    return max_y * exp(-(pow(p.x,2) + pow(p.z,2))/div);
}

// radial crevice on the volcano, one sided
float crevice(const in vec3 p, const in float height, const in float angle)
{
    float stuff = cos(angle) * p.x + sin(angle) * p.z;
    vec2 dir = vec2(cos(angle),sin(angle));
    if(dot(dir, p.zx) > -0.2)
        return height * exp(-1/(1 + 2*max(length(p) - 2,0.))*(pow(stuff, 2)/0.1));
    else
        return 0.;
}

// volcano main shape
float volcano_dome(const in vec3 p)
{
    float max_height2 = 2.;
    vec2 dir = vec2(-0.5,-0.5);
    return terrain_offset + gaussian(p, max_height2, 3.);// + dot(dir, p.xz - vec2(2,2)) / 3.;
}

// volcano with crater, crevice and noise
float terrain_y(const in vec3 p)
{
    float max_height1 = 1.;

    float noise = noise2(p.xz);

    return volcano_dome(p) - gaussian(p, max_height1, 0.1) - crevice(p, 0.6, -3.14/6.) + noise;
}

// lava surface, uses the volcano dome main shape, offset, noise it up, make it "flow"
float lava_y(const in vec3 p)
{
    float noise = 0.2*noise2(p.xz /2. - 0.04 * time * vec2(cos(-3.14/6.),sin(3.14/6.)));
    return volcano_dome(p) + lava_offset + noise;
}

// for ray marching and determining shadows
float terrain_dist(in vec3 p)
{
    return p.y - terrain_y(p);
}

// for red glow on water surface
float lava_dist(in vec3 p)
{
    return crevice(p, 1, -3.14/6.) * max(lava_y(p) - terrain_y(p) + 0.5, 0.);
}

// ray march the scene, only returns whether to shadow or not
bool marchShadow(in vec3 ro, in vec3 rd)
{
    float t = DIST_MIN;

    for(int i=0; i<RAY_MARCH_STEPS, t<=DIST_MAX; ++i) 
    {
        float s = terrain_dist(ro + t*rd);

        if(s<RAY_MARCH_PRECI) 
        {
            //s += t;
            return true;
        }

        t = t + s;
    }

    // marched too long, return background
    return false;
}

// lava color using noise (flows "downward")
vec3 lava_color(in vec3 p)
{
    float color_noise = noise2(p.xz*10. - 0.1 * time * vec2(cos(-3.14/6.),sin(3.14/6.)));
    //color_noise = mix(color_noise, noise2(w_position.yy*5.), 0.5);

    return vec3((lava_depth * 10. + 0.3) * (0.5*color_noise*vec3(1,1,0) + 2.*(max(p.y + 1, 0.))*vec3(1,0,0)));
}

void main() {
    vec3 w_normal_unit = normalize(w_normal);

    // Camera direction for Phong, etc
    vec3 w_camera_dir = normalize(w_position - w_camera_position);

    // irradiance accumulator
    vec4 L = vec4(0,0,0,1);

    // whether terrains casts a shadow or not
    bool shadow = marchShadow(w_position, light_dir);

    // Water is white on top
    if(surface_type == water_type && w_position.y > 0.3 && !shadow)
        L = vec4(1);

    else if(surface_type == smoke_type) // smoke billboard
    {
        // noisy color flowing in 2 directions
        L.rgb = vec3(noise2(frag_tex_coords.xy*4.56 - vec2(0.1,0.7) * time)) + vec3(noise2(frag_tex_coords*6.54 - time * vec2(-0.23,0.451)));
        // alpha is a gaussian in x
        L.w = max(gaussian(frag_tex_coords.xxx - 0.5, 1, 0.1*pow(frag_tex_coords.y,2)) -0.1, 0.);// * noise2(frag_tex_coords * vec2(0.8,0.1) - vec2(time/5.));
    }

    else if(surface_type == skybox_type) // Skybox
        L = vec4(texture(diffuse_map, frag_tex_coords).rgb,1); // read the texture

    else // the rest uses Phong
    {
        L.rgb += k_a; // ambient

        // Terrain Phong shading
        if(surface_type == terrain_type) {
            //hot looking volcano top
            L.rgb += vec3((pow(max(w_position.y - 0.8, 0.), 2.))*vec3(1,0,0));
        }
        
        else if(surface_type == lava_type)
        {
            // lava color noise
            L.rgb += lava_color(w_position); 
        }

        else if(surface_type == water_type)
        {
            // foam effect near rocks, as ambient (even in shadows)
            if(!shadow)
                L.rgb += 0.2 * foam * vec3(1);  

            // red glow effect over lava
            L.rgb += vec3(lava_dist(w_position),0,0);
        }

        else if (surface_type == cloud_type)
        {
            L.rgb += texture(diffuse_map, frag_tex_coords).rgb;
        }

        // no shadow, Phong for diffusive and specular
        if(!shadow) 
        {
            vec3 n = w_normal_unit;

            // normal mapping on water surface to make it less "flat"
            if(surface_type == water_type)
            {
                vec3 normal = texture(normalmap, w_position.xz*2).xzy;
                n = normalize(mix(normal, w_normal, 0.5));
            }
                
            // Phong vectors
            vec3 v = w_camera_dir;
            vec3 l = normalize(light_dir);
            vec3 e = reflect(l,n);
            
            // Phong cosines
            float diff = max(dot(n,l),0.);
            float spec = max(dot(e,v),0.);
            
            // diffusive coef will be edited for some surfaces
            vec3 K_D = k_d;

            // textures on the terrain as diffusive color
            if(surface_type == terrain_type)
            {
                // load 3 textures
                vec3 rock_k_d = texture(rock, w_position.xz/100.).rgb;
                vec3 grass_k_d = texture(grass, w_position.xz/100.).rgb;
                vec3 sand_k_d = texture(sand, w_position.xz/100.).rgb;

                // noise to make the edges less neat
                float tex_noise = noise2((w_position.xz * 100.));

                // use them depending on terrain elevation
                if (w_position.y < 0.7 - tex_noise/1.2)
                    K_D = mix(sand_k_d, grass_k_d, tex_noise);
                else if (w_position.y < 0.7 + tex_noise/3.)
                    K_D = mix(grass_k_d, rock_k_d, 1 - tex_noise);
                else
                    K_D = mix(rock_k_d, grass_k_d, tex_noise);
            }

            // foam effect on water near rocks, as diffusive (only in light)
            if(surface_type == water_type)
                K_D += vec3(1) * foam;

            else if(surface_type == wood_type)
                K_D *= noise2(w_position.xz * 30.);

            // Phong diffusive and specular formula
            L += vec4(diff*K_D + pow(spec, s) * k_s, 0);
        }

        // Distance fog, could use some smooth moving noise...
        L.rgb += (dot(w_camera_dir, vec3(0,-1,0)) + 0.5)/1.5 * (0.5*int(shadow) * w_position.y + 1)/2. * fog_coef * vec3(1) * distance(w_camera_position, w_position) * noise2(vec2(w_position.zx/2.));// * noise2(vec2(time) * length(w_position.xz / 20.) * w_position.xz));

        // water transparency
        if(surface_type == water_type) 
            L.w = 0.7;
        //else if(surface_type == bubble_type)
        //    L.w = 0.9;

    }

    // color the whole scene depending on sunlight color
    L.rgb *= light_color;

    // aaaaaand, we're done
    out_color = L;
}
