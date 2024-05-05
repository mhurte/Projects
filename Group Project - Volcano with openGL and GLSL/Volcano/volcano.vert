#version 330 core

// input attribute variable, given per vertex
in vec3 position;
in vec3 normal;
// output in world coordinates
out vec3 w_position, w_normal;  

// world camera position
uniform vec3 w_camera_position;

// texturing
in vec2 tex_coord;
out vec2 frag_tex_coords;

// global matrix variables
uniform mat4 model;
uniform mat4 view;
uniform mat4 skybox_view;
uniform mat4 billboard_view;
uniform mat4 projection;

// time
uniform float time;

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

// offsets for surfaces
uniform float terrain_offset;
uniform float water_offset;
uniform float lava_offset;

// directional light
uniform vec3 light_dir;

// cacti
uniform float draw_cactus;

// used in generating perlin noise
const int N_perlin = 10;
#define PERLIN_SCALE 3
#define PERLIN_ITER 4

// ray marching constants
#define DIST_MIN 1e-2
#define DIST_MAX 1e2
#define RAY_MARCH_STEPS 100
#define RAY_MARCH_PRECI 1e-3

// noise at given vertex
out float noise;

// whether to light or not (ray marched)
//out float shadow; // used as a bool (<0.5 <=> false)

// foam effect on water
out float foam;

// depth to lava for water effect, etc
out float lava_depth;

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
    vec2 dir = vec2(1,1);
    if(dot(dir, p.xz) > -0.)
        return height * exp(-1/(1 + 2*max(length(p.xz) - 2,0.))*(pow(stuff, 2)/0.1));
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

    noise = noise2(p.xz);

    float volcano = volcano_dome(p) - gaussian(p, max_height1, 0.1) - crevice(p, 0.6, -3.14/6.) + noise;

    float bridge = (1+length(p)/10.) * crevice(p.zyx, 0.4, 0.);

    return volcano + bridge;
}

// volcano normal
vec3 terrain_normal(const in vec3 p)
{
  float eps = 0.001;
  vec3 epsx = vec3(eps, 0, 0);
  vec3 epsz = vec3(0, 0, eps);
  vec3 n = vec3(terrain_y(p-epsx) - terrain_y(p+epsx), 
                2. * eps, 
                terrain_y(p-epsz) - terrain_y(p+epsz));
  return normalize(n);
}

// pseudo random waves
float waves(const in vec3 p, const float max_y)
{
    noise = noise2(p.xz/4. + 0.1 * time * vec2(0.2,0.4));
    return max_y * noise;//max_y/2 * mix(cos((p.x + rand(p.xx)/2.) * 10 + time + rand(p.xz)) + sin((p.z + rand(p.zz)/2.) * 10 + time + rand(p.xz)), 2 * rand(p.xz), 0.8) + max_y;
}

// water surface
float water_y(const in vec3 p)
{
    return water_offset + waves(p, 0.5);
}

// water normal
vec3 water_normal(const in vec3 p)
{
  float eps = 0.001;
  vec3 epsx = vec3(eps, 0, 0);
  vec3 epsz = vec3(0, 0, eps);
  vec3 n = vec3(water_y(p-epsx) - water_y(p+epsx), 
                2. * eps, 
                water_y(p-epsz) - water_y(p+epsz));
  return normalize(n);
}

// lava surface, uses the volcano dome main shape, offset, noise it up, make it "flow"
float lava_y(const in vec3 p)
{
    noise = 0.2*noise2(p.xz /2. - 0.04 * time * vec2(cos(-3.14/6.),sin(3.14/6.)));
    return volcano_dome(p) + lava_offset + noise;
}

// lava normal
vec3 lava_normal(const in vec3 p)
{
  float eps = 0.001;
  vec3 epsx = vec3(eps, 0, 0);
  vec3 epsz = vec3(0, 0, eps);
  vec3 n = vec3(lava_y(p-epsx) - lava_y(p+epsx), 
                2. * eps, 
                lava_y(p-epsz) - lava_y(p+epsz));
  return normalize(n);
}

// for ray marching and determining shadows
//float scene(in vec3 p)
//{
//    return p.y - terrain_y(p);
//}
//
//// ray march the scene, only returns whether to shadow or not
//float marchShadow(in vec3 ro, in vec3 rd)
//{
//    float t = DIST_MIN;
//
//    for(int i=0; i<RAY_MARCH_STEPS, t<=DIST_MAX; ++i) 
//    {
//        float s = scene(ro + t*rd);
//
//        if(s<RAY_MARCH_PRECI) 
//        {
//            //s += t;
//            return 1.;
//        }
//
//        t = t + s;
//    }
//
//    // marched too long, return background
//    return 0.;
//}

// Apply quaternion q to vector v
vec3 rotate(vec3 v, vec4 q)
{ 
	return v + 2.0 * cross(cross(v, q.xyz) + v*q.w, q.xyz);
} 

void main() {
    w_position = (model * vec4(position, 1)).xyz; // go to world coordinates

    if(surface_type == terrain_type) // terrain
    {
        w_normal = terrain_normal(w_position); // recompute normals
        w_position = vec3(w_position.x, terrain_y(w_position), w_position.z); // height field
    }

    else if(surface_type == water_type) // water
    {
        w_normal = water_normal(w_position); // recompute normals
        w_position = vec3(w_position.x, water_y(w_position), w_position.z); // height field

        // foam effect near rocks depends on height diff
        float diff_terrain = w_position.y - terrain_y(w_position);
        if(0. < diff_terrain && diff_terrain < 0.1)
            foam = 10.*(0.1 - diff_terrain);
        else
            foam = 0.;

        // lava depth under water (if lava over terrain) to make the water glow
        lava_depth = lava_y(w_position) > terrain_y(w_position) ? max(w_position.y - lava_y(w_position), 0.) : 1e4;
    }

    else if(surface_type == lava_type) // lava
    {
        w_normal = lava_normal(w_position); // recompute normals
        w_position = vec3(w_position.x, lava_y(w_position), w_position.z); // height field

        // lava depth over terrain (to make the edges darker)
        lava_depth = max(w_position.y - terrain_y(w_position), 0.);
    }

    else if(surface_type == wood_type )
    {
        w_position += vec3(0, water_y(w_position), 0);

        vec3 normal = water_normal(w_position);
        vec4 q = vec4(-cross(vec3(0,1,0), normal), dot(vec3(0,1,0), normal));
        w_normal = rotate((model * vec4(normal, 0)).xyz, q);
    }

    else if(surface_type == cloud_type)
    {
        w_position += vec3(0, water_y(w_position), 0);

        w_normal = (model * vec4(normal, 0)).xyz;        
    }

    else if(surface_type == cactus_type)
    {
        if(draw_cactus > 0.5)
            w_position += vec3(0, volcano_dome(w_position) + 0.3, 0);
        else
            w_position = vec3(0,1000,0);
        w_normal = (model * vec4(normal, 0)).xyz;
    }
    else
    {
        w_normal = (model * vec4(normal, 0)).xyz; // normal in world coordinates
    }

    if(surface_type == skybox_type) // skybox
    {
        // Non translated view matrix and homogeneous coefficient rescaling (to put it far away)
        gl_Position = (projection * skybox_view * model * vec4(position, 1./30.));  
    }
    else if(surface_type == smoke_type) // smoke billboard
    {
        // cylindrical view model matrix
        mat4 modelview = view * model;
        
        modelview[0][0] = 1.0; 
        modelview[0][1] = 0.0; 
        modelview[0][2] = 0.0; 

        modelview[2][0] = 0.0; 
        modelview[2][1] = 0.0; 
        modelview[2][2] = 1.0;  

        w_position = (modelview * vec4(position, 1)).xyz;

        gl_Position = projection * modelview * vec4(position, 1);
    }
    else
    {
        gl_Position = projection * view * vec4(w_position, 1);  // usual
    }

    // pass the textures coordinates to fragment shader
    frag_tex_coords = tex_coord;

    // No shadows on some surface
    //if(surface_type == smoke_type || surface_type == projection_type || surface_type == skybox_type)
    //    shadow = 0.;
    //else 
    //    shadow = marchShadow(w_position, normalize(light_dir)); // else, ray march in light dir and check if terrain casts a shadow
    
}
